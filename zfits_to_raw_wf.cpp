//Prerequisite: have CamerasToACTL compiled on the machine
//Go to the root of the CamerasToACTL project ()
//Activate root: source ~/software/root/build_mine/bin/thisroot.sh
//Add the library path to your environment: export LD_LIBRARY_PATH=<path_to_cam2actl_root>/Build.Release/lib:$LD_LIBRARY_PATH
//Compile me with: 
//$ ACTLROOT="/home/ctauser/software/CamerasToACTL/trunk"
//$ g++ -O3 -o zfits_to_raw_wf zfits_to_raw_wf.cpp -I${ACTLROOT}/IO/Fits -I${ACTLROOT}/Core -I${ACTLROOT}/Build.Release/Core -L${ACTLROOT}/Build.Release/lib -std=c++11 -lACTLCore -lZFitsIO -lprotobuf $(root-config --libs) -I/opt/tensorflow/bazel-tensorflow -L/opt/tensorflow/bazel-bin/tensorflow/ -ltensorflow_cc
//use zfits reader
#include "/home/ctauser/software/CamerasToACTL/trunk/IO/Fits/ProtobufIFits.h"
//use generic arrays helpers
#include "AnyArrayHelper.h"
//use CameraEvents
#include "L0.pb.h"
//standard stuff
#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <functional>
#include <cstdlib>
// ROOT stuff
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TCanvas.h"
#include <TH2.h>
#include <TStyle.h>
#include <TString.h>
#include <TList.h>
#include <TLine.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TSystem.h>
#include <TF1.h>
#include <TDatime.h>
#include "TGraphErrors.h"
#include "TVirtualFFT.h"
// TensorFlow stuff
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using namespace ACTL;
using namespace ACTL::IO;
using namespace ACTL::AnyArrayHelper;
using namespace DataModel;
using namespace std;


void get_correlation(
        const UShort_t wf1[], const UShort_t wf2[], const UInt_t num_samples,
        const std::vector<Int_t> delays_sample,
        std::vector<ULong64_t> &size_delays,
        std::vector<Double_t> &sum_wf1_delays,
        std::vector<Double_t> &sum_wf2_delays,
        std::vector<Double_t> &sum_wf11_delays,
        std::vector<Double_t> &sum_wf12_delays,
        std::vector<Double_t> &sum_wf22_delays )
{
  for(UInt_t d_i=0; d_i<delays_sample.size(); d_i++){
    Int_t delay_sample = delays_sample[d_i];
    if(delay_sample<0){
      for(Int_t sample_wf1=0;sample_wf1<(num_samples+delay_sample);sample_wf1++){
        Int_t sample_wf2=sample_wf1 - delay_sample;
        size_delays[d_i]+=1;
        UShort_t wf1_sample=wf1[sample_wf1];
        UShort_t wf2_sample=wf2[sample_wf2];
        sum_wf1_delays[d_i]+=wf1_sample;
        sum_wf2_delays[d_i]+=wf2_sample;
        sum_wf11_delays[d_i]+=wf1_sample*wf1_sample;
        sum_wf12_delays[d_i]+=wf1_sample*wf2_sample;
        sum_wf22_delays[d_i]+=wf2_sample*wf2_sample;
      }
    }
    if(delay_sample>=0){
      for(Int_t sample_wf2=0;sample_wf2<(num_samples-delay_sample);sample_wf2++){
        Int_t sample_wf1=sample_wf2 + delay_sample;
        size_delays[d_i]+=1;
        UShort_t wf1_sample=wf1[sample_wf1];
        UShort_t wf2_sample=wf2[sample_wf2];
        sum_wf1_delays[d_i]+=wf1_sample;
        sum_wf2_delays[d_i]+=wf2_sample;
        sum_wf11_delays[d_i]+=wf1_sample*wf1_sample;
        sum_wf12_delays[d_i]+=wf1_sample*wf2_sample;
        sum_wf22_delays[d_i]+=wf2_sample*wf2_sample;
      }
    }
  }
}


int main(int argc, char** argv) {
  if ((argc > 7) || (argc < 4)) {
    cout<<"usage: "<<argv[0]<<" path_data n_file start zfits_prefix [model baseline1 baseline2]"<<endl;
    return 1;
  }

  TString suffBase = "_raw_waveforms";

  string path = string(argv[1]);
  int nfiles  = atoi(argv[2]);
  int start   = atoi(argv[3]);
  string suff = string(argv[4]);
  sting model = "";
  if (argc > 5){
    model = string(argv[5]);
    baseline_pix1 = string(argv[6]);
    baseline_pix2 = string(argv[7]);
  }
  UInt_t max_delay_sample = 500;
  float sample_ns = 4;
  //creating the output file to store reprocessed data

  TFile *fout = new TFile(Form("%s_%04.f_%04.f%s.root",suff.data(),(double)start,(double)(start+nfiles-1),suffBase.Data()),"recreate");
  fout->cd();

  TTree *waveforms = new TTree("waveforms","Tree with raw waveforms");
  UInt_t num_samples = 4320;
  UShort_t wf1[num_samples];
  UShort_t wf2[num_samples];
  UShort_t t_s;
  UShort_t t_ns;
  UInt_t evt_nr;
  waveforms->Branch("wf1", &wf1,Form("wf1[%d]/s", num_samples));
  waveforms->Branch("wf2", &wf2,Form("wf2[%d]/s", num_samples));
  waveforms->Branch("time_s", &t_s,"time in s/s");
  waveforms->Branch("time_ns", &t_ns,"time in ns/s");
  waveforms->Branch("event_number", &evt_nr,"event number/i");

  std::vector<Int_t> delays_sample;
  std::vector<Double_t> size_delays, delays_ns;
  std::vector<Double_t> sum_wf1_delays, sum_wf2_delays, sum_wf11_delays, sum_wf12_delays, sum_wf22_delays;

  for(Int_t delay_sample=-max_delay_sample; delay_sample<=(Int_t)max_delay_sample; delay_sample++){
    delays_sample.push_back(delay_sample);
    delays_ns.push_back(delay_sample * sample_ns);
    size_delays.push_back(0);
    sum_wf1_delays.push_back(0);
    sum_wf2_delays.push_back(0);
    sum_wf11_delays.push_back(0);
    sum_wf12_delays.push_back(0);
    sum_wf22_delays.push_back(0);
  }

  for(int ifile=0; ifile<nfiles; ifile++) {
    float fileno=float(start+ifile);
    string infile = Form("%s/%s_%04.f.fits.fz",path.data(),suff.data(),fileno);
    int count = 0;
    Printf("opening %s",infile.data());

    //Open the desired input file
    ProtobufIFits input_file(infile, "Events");

    //loop over all the events in the file
    Printf("file %d, Events = %d",ifile,input_file.getNumMessagesInTable());
    CameraEvent* event = input_file.readTypedMessage<CameraEvent>(1);
    if (num_samples != getNumElems(event->sii().samples())){
        Printf("Expected %d samples but got %d", num_samples, getNumElems(event->sii().samples()));
        return -1;
    }

    std::vector<CameraEvent*> eventsCh0;
    std::map<uint64, CameraEvent*> eventsCh1;
    
    Printf("Splitting events in Ch0 and Ch1");
    for (uint32 i=0;i<input_file.getNumMessagesInTable();i++) {
      
      event = input_file.readTypedMessage<CameraEvent>(i+1);
      uint32 event_nr = event->eventnumber();
      int pix = event->sii().channelid();
      if(pix==0) eventsCh0.push_back(event);
      else eventsCh1[event_nr] = event;
    }

    Printf("Associating events on both channels and calculating g2");
    uint64 synchro_event=0;
    uint64 t0_s, event0_nr;
    for(UInt_t i=0; i<eventsCh0.size(); i++) {
      uint32 event_nr = eventsCh0[i]->eventnumber();
      uint64 time_s = eventsCh0[i]->local_time_sec();
      uint64 time_ns = eventsCh0[i]->local_time_nanosec();
      if(eventsCh1[event_nr]) {
        synchro_event++;
        if (synchro_event==1){
          t0_s = time_s;
          event0_nr = event_nr;
        }
        if((synchro_event%1000)==1){
          Printf(
            "event %d: t1=%d.%09d s, t2=%d.%09d s", 
            synchro_event, time_s - t0_s, time_ns, 
            eventsCh1[event_nr]->local_time_sec() - t0_s, 
            eventsCh1[event_nr]->local_time_nanosec()
          );
        }
        const int16* waveformsPix0 = readAs<int16>(eventsCh0[i]->sii().samples());
        const int16* waveformsPix1 = readAs<int16>(eventsCh1[event_nr]->sii().samples());
        Double_t mean_wf1=0;
        Double_t mean_wf2=0;
        for(int ip=0; ip<num_samples; ip++){
          wf1[ip] = (UShort_t)waveformsPix0[ip];
          wf2[ip] = (UShort_t)waveformsPix1[ip];
          mean_wf1 += wf1[ip];
          mean_wf2 += wf2[ip];
        }
        mean_wf1 /= num_samples;
        mean_wf2 /= num_samples;
        t_ns = (UShort_t) time_ns;
        t_s = (UShort_t) time_s;
        evt_nr = (UInt_t) event_nr;
        waveforms->Fill();
        count++;
        get_correlation(
          wf1, wf2, delays_sample, num_samples, size_delays, sum_wf1_delays, sum_wf2_delays,
          sum_wf11_delays, sum_wf12_delays, sum_wf22_delays
        )
      }
    }
    Printf("%d/%d(Ch0)/%d(Ch1) associated samples",count,eventsCh0.size(),eventsCh1.size());
  }

  std::vector<Double_t> g2_11, g2_12, g2_22;
  for(UInt_t d_i=0; d_i<delays_sample.size(); d_i++){
    g2_11.push_back(
      size_delays[d_i]*sum_wf11_delays[d_i]/(sum_wf1_delays[d_i]*sum_wf1_delays[d_i])
    );
    g2_12.push_back(
      size_delays[d_i]*sum_wf12_delays[d_i]/(sum_wf1_delays[d_i]*sum_wf2_delays[d_i])
    );
    g2_22.push_back(
      size_delays[d_i]*sum_wf22_delays[d_i]/(sum_wf2_delays[d_i]*sum_wf2_delays[d_i])
    );
  }
  TGraph *num_samples_gr = new TGraph(delays_ns.size(), &delays_ns[0], &size_delays[0]);
  num_samples_gr->SetTitle("number of samples;delay [ns];number of samples");
  num_samples_gr->SetName("num_samples_gr");
  TGraph *sum_wf1_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf1_delays[0]);
  sum_wf1_gr->SetTitle("sum_wf1;delay [ns];sum of waveform 1");
  sum_wf1_gr->SetName("sum_wf1_gr");
  TGraph *sum_wf2_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf2_delays[0]);
  sum_wf2_gr->SetTitle("sum_wf2;delay [ns];sum of waveform 2");
  sum_wf2_gr->SetName("sum_wf2_gr");
  TGraph *sum_wf11_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf11_delays[0]);
  sum_wf11_gr->SetTitle("sum_wf11;delay [ns];sum of squared waveform 1");
  sum_wf11_gr->SetName("sum_wf11_gr");
  TGraph *sum_wf12_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf12_delays[0]);
  sum_wf12_gr->SetTitle("sum_wf12;delay [ns];sum of (waveform 1 * waveform 2)");
  sum_wf12_gr->SetName("sum_wf12_gr");
  TGraph *sum_wf22_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf22_delays[0]);
  sum_wf22_gr->SetTitle("sum_wf22;delay [ns];sum of squared waveform 2");
  sum_wf22_gr->SetName("sum_wf22_gr");
  TGraph *g2_11_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_11[0]);
  g2_11_gr->SetTitle("g2_11;delay [ns];g2_11");
  g2_11_gr->SetName("g2_11_gr");
  TGraph *g2_12_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_12[0]);
  g2_12_gr->SetTitle("g2_12;delay [ns];g2_12");
  g2_12_gr->SetName("g2_12_gr");
  TGraph *g2_22_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_22[0]);
  g2_22_gr->SetTitle("g2_22;delay [ns];g2_22");
  g2_22_gr->SetName("g2_22_gr");
  UShort_t wf1_min = waveforms->GetMinimum("wf1");
  UShort_t wf1_max = waveforms->GetMaximum("wf1");
  UShort_t wf2_min = waveforms->GetMinimum("wf2");
  UShort_t wf2_max = waveforms->GetMaximum("wf2");
  TH1F * wf1_hist = new TH1F("wf1_hist", "raw ADC histogram for waveform 1", wf1_max-wf1_min, wf1_min, wf1_max);
  waveforms->Draw("wf1>>wf1_hist");
  TH1F * wf2_hist = new TH1F("wf2_hist", "raw ADC histogram for waveform 2", wf2_max-wf2_min, wf2_min, wf2_max);
  waveforms->Draw("wf2>>wf2_hist");
  fout->cd();
  waveforms->Write("", TObject::kOverwrite);
  Printf("writing graphs");
  num_samples_gr->Write();
  sum_wf1_gr->Write();
  sum_wf2_gr->Write();
  sum_wf11_gr->Write();
  sum_wf12_gr->Write();
  sum_wf22_gr->Write();
  g2_11_gr->Write();
  g2_12_gr->Write();
  g2_22_gr->Write();
  wf1_hist->Write();
  wf2_hist->Write();
  Printf("closing file");
  fout->Close();
}

