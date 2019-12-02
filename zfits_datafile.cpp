#include "zfits_datafile.h"
// standard stuff
#include <iostream>
#include <map>
#include <numeric>
#include <functional>
#include <cstdlib>
// ROOT stuff
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TString.h>
#include <TF1.h>
// zfits stuff
#include "ProtobufIFits.h"
#include "AnyArrayHelper.h"
#include "L0.pb.h"

#include "root_datafile.h"

using namespace ACTL;
using namespace ACTL::IO;
using namespace ACTL::AnyArrayHelper;
using namespace DataModel;
using namespace std;

void zfits_to_root(
  const char output_filename[], const vector<string> zfits_files,
  const unsigned long max_delay_sample, const float sample_ns, 
  const size_t n_sample, const Long64_t n_waveform
) {
  // Create tree
  TFile *fout = new TFile(output_filename, "recreate");
  fout->cd();
  TTree *waveforms = new TTree("waveforms","Tree with raw waveforms");
  UShort_t wf1[n_sample];
  TBranch * wf1_br = waveforms->Branch("wf1", &wf1,Form("wf1[%zu]/s", n_sample));
  UShort_t wf2[n_sample];
  TBranch * wf2_br = waveforms->Branch("wf2", &wf2,Form("wf2[%zu]/s", n_sample));
  UShort_t t_s;
  TBranch * t_s_br = waveforms->Branch("time_s", &t_s,"time in s/s");
  UShort_t t_ns;
  TBranch * t_ns_br = waveforms->Branch("time_ns", &t_ns,"time in ns/s");
  UInt_t evt_nr;
  TBranch * evt_nr_br = waveforms->Branch("event_number", &evt_nr,"event number/i");

  // Loop on files
  for(size_t ifile=0; ifile<zfits_files.size(); ifile++) {
    // Open the desired input file
    cout<<"opening "<<zfits_files[ifile]<<endl;
    ProtobufIFits input_file(zfits_files[ifile].c_str(), "Events");
    cout<<"file "<<ifile<<", Events = "<<input_file.getNumMessagesInTable()<<endl;

    // Check the number of samples
    CameraEvent* event = input_file.readTypedMessage<CameraEvent>(1);
    if (n_sample != getNumElems(event->sii().samples())){
        cout<<"Expected "<<n_sample<<" samples but got "<<getNumElems(event->sii().samples())<<endl;
        return;
    }

    // Loop over the events from both pizels in the file
    std::vector<CameraEvent*> eventsCh0;
    std::map<uint64, CameraEvent*> eventsCh1;    
    cout<<"Splitting events in Ch0 and Ch1"<<endl;
    for (uint32 i=0;i<input_file.getNumMessagesInTable();i++) {
      event = input_file.readTypedMessage<CameraEvent>(i+1);
      uint32 event_nr = event->eventnumber();
      int pix = event->sii().channelid();
      if(pix==0) eventsCh0.push_back(event);
      else eventsCh1[event_nr] = event;
    }

    // Lopp over events with data from both pixels
    cout<<"Associating times in Ch0 and Ch1"<<endl;
    size_t synchro_event=0;
    uint64 t0_s=0, t0_ns=0;
    for(UInt_t i=0; i<eventsCh0.size(); i++) {
      uint32 event_nr = eventsCh0[i]->eventnumber();
      uint64 time_s = eventsCh0[i]->local_time_sec();
      uint64 time_ns = eventsCh0[i]->local_time_nanosec();
      auto search = eventsCh1.find(event_nr);
      if(search == eventsCh1.end()) {
        continue;  // skip event if 2nd pixel has no data
      }
      synchro_event++;
      if (synchro_event==1){
        t0_s = time_s;
        t0_ns = time_ns;        
      }
      if((synchro_event%1000)==1){
        Long64_t dt1_s = time_s - t0_s;
        Long64_t dt1_ns = time_ns - t0_ns;
        Long64_t dt2_s = eventsCh1[event_nr]->local_time_sec() - t0_s;
        Long64_t dt2_ns = eventsCh1[event_nr]->local_time_nanosec() - t0_ns;
        if (dt1_ns<0) {
            dt1_ns += 1000000000;
            dt1_s -= 1;
            dt2_ns += 1000000000;
            dt2_s -= 1;
        }
        Printf(
          "event %zu: t1=%lli.%09lli s, t2=%lli.%09lli s", 
          synchro_event, dt1_s, dt1_ns, dt2_s, dt2_ns
        );
      }
      const int16* waveformsPix0 = readAs<int16>(eventsCh0[i]->sii().samples());
      const int16* waveformsPix1 = readAs<int16>(eventsCh1[event_nr]->sii().samples());
      for(size_t ip=0; ip<n_sample; ip++){
        wf1[ip] = (UShort_t)waveformsPix0[ip];
        wf2[ip] = (UShort_t)waveformsPix1[ip];
      }
      t_ns = (UShort_t) time_ns;
      t_s = (UShort_t) time_s;
      evt_nr = (UInt_t) event_nr;
      waveforms->Fill();
      if( (n_waveform > 0) && (synchro_event >= size_t(n_waveform)) ) {
        cout<<"we converted "<<n_waveform<<" waveforms"<<endl;
        break;
      }
    }
    cout<<synchro_event<<" events synchronized / "<<eventsCh0.size()<<"(Ch0) / ";
    cout<<eventsCh1.size()<<"(Ch1)"<<endl;
  }
  // Write to datafile
  wf1_br->ResetAddress();
  wf2_br->ResetAddress();
  t_s_br->ResetAddress();
  t_ns_br->ResetAddress();
  evt_nr_br->ResetAddress();
  waveforms->Write("", TObject::kOverwrite);
  fout->Close(); //Close delete the tree
  delete fout;
}

