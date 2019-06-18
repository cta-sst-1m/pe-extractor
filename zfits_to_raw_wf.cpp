//Prerequisite: have CamerasToACTL compiled on the machine
//Go to the root of the CamerasToACTL project ()
//Activate root: source /usr/local/bin/thisroot.sh
//Add the library path to your environment: export LD_LIBRARY_PATH=<path_to_cam2actl_root>/Build.Release/lib:$LD_LIBRARY_PATH
//Compile me with: 
//$ ACTLROOT="/home/ctauser/software/CamerasToACTL/trunk"
//$ g++ -o zfits_to_raw_wf zfits_to_raw_wf.cpp -I${ACTLROOT}/IO/Fits -I${ACTLROOT}/Core -I${ACTLROOT}/Build.Release/Core -L${ACTLROOT}/Build.Release/lib -std=c++11 -lACTLCore -lZFitsIO -lprotobuf $(root-config --libs)
//use zfits reader
#include "/home/ctauser/software/CamerasToACTL/trunk/IO/Fits/ProtobufIFits.h"

//use generic arrays helpers
#include "AnyArrayHelper.h"

//use CameraEvents
#include "L0.pb.h"

#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <functional>
#include <cstdlib>

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

using namespace ACTL;
using namespace ACTL::IO;
using namespace ACTL::AnyArrayHelper;
using namespace DataModel;
using namespace std;


int main(int argc, char** argv) {

  TString suffBase = "_raw_waveforms";

  string path = string(argv[1]);
  int nfiles  = atoi(argv[2]);
  int start   = atoi(argv[3]);
  string suff = string(argv[4]);

  //creating the output file to store reprocessed data
  uint32 num_samples = 4320;
  Double_t wf1[num_samples];
  Double_t wf2[num_samples];
  
  TFile *fout = new TFile(Form("SST1M_01_%s_%04.f_%04.f%s.root",suff.data(),(double)start,(double)(start+nfiles-1),suffBase.Data()),"recreate");
  fout->cd();
  TTree *outTr = new TTree("waveforms","Tree with raw waveforms");
  outTr->Branch("wf1",&wf1,Form("wf1[%d]/D", num_samples));
  outTr->Branch("wf2",&wf2,Form("wf2[%d]/D", num_samples));

  int nbins = 300000;
  double upx = (double)nbins;
  

  for(int ifile=0; ifile<nfiles; ifile++) {
    float fileno=float(start+ifile);
    string infile = Form("%s/SST1M_01_%s_%04.f.fits.fz",path.data(),suff.data(),fileno);
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
    std::map<int, CameraEvent*> eventsCh1; 
    
    Printf("Splitting events in Ch0 and Ch1");
    for (uint32 i=0;i<input_file.getNumMessagesInTable();i++) {
      
      event = input_file.readTypedMessage<CameraEvent>(i+1);
      int pix  = event->sii().channelid();
      int time = event->local_time_nanosec();
      if(pix==0) eventsCh0.push_back(event);
      else eventsCh1[time] = event;
    }

    Printf("Associating times in Ch0 and Ch1");
    for(int i=0; i<eventsCh0.size(); i++) {
      if((i%500000)==0)Printf("event %d",i);
      int timeCh0 = eventsCh0[i]->local_time_nanosec();
      if(eventsCh1[timeCh0]) {
	const int16* waveformsPix0 = readAs<int16>(eventsCh0[i]->sii().samples());
	const int16* waveformsPix1 = readAs<int16>(eventsCh1[timeCh0]->sii().samples());
	
	for(int ip=0; ip<num_samples; ip++){
	  wf1[ip] = (Double_t)waveformsPix0[ip];
	  wf2[ip] = (Double_t)waveformsPix1[ip];
	}
        outTr->Fill();
        count++;
      }
    }
    Printf("%d/%d(Ch0)/%d(Ch1) associated samples",count,eventsCh0.size(),eventsCh1.size());
  }
  
  fout->cd();
  Printf("writing tree and closing file");
  outTr->Write();
  fout->Close();
}

