#include <getopt.h>
#include <iostream>

#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>

#include "zfits_datafile.h"
#include "root_datafile.h"
#include "extract_pe.h"

#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;
namespace tensorflow{ class Session; }

void usage(){
  cout<<"usage:"<<endl;
  cout<<"\tanalyze_SIIS [options] <model> <zfits_files>"<<endl;
  cout<<"options:"<<endl;
  cout<<"\t--output <ROOTFILE>:         Name of the root file to create. [Default: waveforms.root]"<<endl;
  cout<<"\t--n_waveform <INT>:          Number of waveform to analyze. Set to <0 values to disable this limit. [Default: -1]"<<endl;
  cout<<"\t--sample_ns <FLOAT>:         Duration of a waveform sample in ns. [Default: 4.]"<<endl;
  cout<<"\t--pe_bin_ns <FLOAT>:         Duration of a bin in ns for the reconstructed photo-electrons. [Default: 0.5]"<<endl;
//  cout<<"\t--n_sample <INT>:            Number of samples in a waveform. [Default: 4320]"<<endl;
  cout<<"\t--batch_size <INT>:          Number of waveforms analyzed together (reduce to decrease use of memory). [Default: 10]"<<endl;
  cout<<"\t--max_delay_sample <INT>:    The calcualtion of g2 with waveforms will be done for delays from -max_delay_sample to max_delay_sample. [Default: 500]"<<endl;
  cout<<"\t--max_delay_bin <INT>:       The calcualtion of g2 from reconstructed pe will be done for delays from -max_delay_bin to max_delay_bin. [Default: 500]"<<endl;
  cout<<"\t--n_cpu <INT>:               Maximum number of CPU to use. [Default: 4]"<<endl;
  cout<<"\t--baseline_margin <FLOAT>:   The threshold for baseline calculation is taken as the minimum of the waveform + baseline_margin. Any sample above that value (and the neigbourghs according to baseline_s_around) are discarded from the baseline estimation. [Default: 7.]"<<endl;
  cout<<"\t--baseline_s_around <INT>: Number of samples discarded around the samples which have a value over the threshold during baseline estimation. [Default: 25]"<<endl;
  cout<<"\t--baseline_n_waveform <INT>: Number of waveform used for baseline estimation. [Default: 1000]"<<endl;
}


int main(int argc, char** argv) {
  // Default values for options
  string output="waveforms.root";
  int input_n_waveform =-1; //no limit on the number of waveform to analyze
  Double_t sample_ns = 4.;
  Double_t pe_bin_ns = .5;
  //size_t n_sample = 4320;
  const size_t n_sample = 4320; //need to be fixed for template function calculate_g2()
  const size_t num_bins_pe = 34560;
  size_t batch_size = 10;
  UInt_t max_delay_sample = 500;
  UInt_t max_delay_bin = 500;
  UInt_t n_cpu_core_to_use = 4;
  Double_t baseline_margin_lsb=7.;
  UInt_t baseline_sample_around=25;
  Long64_t baseline_n_waveform=1000;
  vector<string> zfits_files;

  // process options
  const char* const short_opts = "o:n:T:t:s:b:D:d:c:m:a:B:h";
  const struct option long_opts[] = {
          {"output", required_argument, nullptr, 'o'},
          {"n_waveform", required_argument, nullptr, 'n'},
          {"sample_ns", required_argument, nullptr, 'T'},
          {"pe_bin_ns", required_argument, nullptr, 't'},
  //        {"n_sample", required_argument, nullptr, 's'},
          {"batch_size", required_argument, nullptr, 'b'},
          {"max_delay_sample", required_argument, nullptr, 'D'},
          {"max_delay_bin", required_argument, nullptr, 'd'},
          {"n_cpu", required_argument, nullptr, 'c'},
          {"baseline_margin", required_argument, nullptr, 'm'},
          {"baseline_s_around", required_argument, nullptr, 'a'},
          {"baseline_n_waveform", required_argument, nullptr, 'B'},
          {"help", no_argument, nullptr, 'h'},
          {nullptr, no_argument, nullptr, 0}
  };
  while (true){
    const auto opt = getopt_long (argc, argv, short_opts, long_opts, NULL);
    if (-1 == opt)
      break;
    switch (opt)
    {
      case 'o': //output
        output = optarg;
        break;
      case 'n': //n_waveform
        input_n_waveform = atoi(optarg);
        break;
      case 'T': //sample_ns
        sample_ns = atof(optarg);
        break;
      case 't': //pe_bin_ns
        pe_bin_ns = atof(optarg);
        break;
/*      case 's': //n_sample
        n_sample = atoi(optarg);
        break;*/
      case 'b': //batch_size
        batch_size = atoi(optarg);
        break;
      case 'D': //max_delay_sample
        max_delay_sample = atoi(optarg);
        break;
      case 'd': //max_delay_bin
        max_delay_bin = atoi(optarg);
        break;
      case 'c': //n_cpu
        n_cpu_core_to_use = atoi(optarg);
        break;
      case 'm': //baseline_margin
        baseline_margin_lsb = atof(optarg);
        break;
      case 'a': //baseline_s_around
        baseline_sample_around = atoi(optarg);
        break;
      case 'B': //baseline_n_waveform
        baseline_n_waveform = atoi(optarg);
        break;
      case 'h': // -h or --help
      case '?': // Unrecognized option
      default:
        usage();
        return -1;
    }
  }
  if (argc-optind <2) {
    cout<<"ERROR: at least a model and a zfits file must be passed as argument."<<endl;
    usage();
    return -1;
  }
  const string model=argv[optind++];
  if ( !file_exists(model) ) {
    cout<<"could not find model file "<<model<<endl;
    return 1;
  }
  cout<<"model: "<<model<<endl;
  while (optind < argc)
  {
    string waveform_file=argv[optind++];
    cout<<"file to analyse: "<<waveform_file<<endl;
    if ( !file_exists(waveform_file) ) {
      cout<<"WARNING: datafile "<<waveform_file<<" not found."<<endl;
      continue;
    }
    zfits_files.push_back(waveform_file);
  }
  //const UInt_t num_bins_pe_per_sample = ((UInt_t) (sample_ns/pe_bin_ns));
  //const size_t num_bins_pe = num_bins_pe_per_sample * n_sample;

  //convert waveforms from zfits to ROOT datafile
  cout<<"convert wavform to ROOT file"<<endl;
  zfits_to_root(output.c_str(), zfits_files, max_delay_sample, sample_ns);

  // open the root file and get data branches
  TFile *f = new TFile(output.c_str(), "update");
  f->cd();
  TTree *waveforms = dynamic_cast<TTree*>(f->Get("waveforms"));
  TBranch *wf1_br = waveforms->GetBranch("wf1");
  TBranch *wf2_br = waveforms->GetBranch("wf2");

  // get baseline & rate
  cout<<"determine baselines cutting "<<baseline_sample_around
    <<" samples around all samples which are above min + "
    <<baseline_margin_lsb<<" LSB."<<endl;
  Double_t baseline_wf1 = get_baseline(
    wf1_br, baseline_margin_lsb, baseline_sample_around, baseline_n_waveform, 
    n_sample
  );
  Double_t baseline_wf2 = get_baseline(
    wf2_br, baseline_margin_lsb, baseline_sample_around, baseline_n_waveform, 
    n_sample
  );
  cout<<"baseline: pixel1="<<baseline_wf1<<"LSB, pixel2="<<baseline_wf2<<" LSB"<<endl;
  Double_t rate_wf1 = get_pe_rate_MHz(wf1_br, baseline_wf1);
  Double_t rate_wf2 = get_pe_rate_MHz(wf2_br, baseline_wf2);
  cout<<"pe rate: pixel1="<<rate_wf1<<" MHz, pixel2="<<rate_wf2<<" MHz"<<endl;

  //make histograms of ADC counts
  cout<<"create ADC count histograms"<<endl;
  histogram_waveform(waveforms, "wf1", baseline_wf1);
  histogram_waveform(waveforms, "wf2", baseline_wf2);

  // add example of waveforms to datafile
  cout<<"create plot of first waveforms."<<endl;
  TGraph* wf1_gr = plot_waveform<UShort_t>(wf1_br, 0, "wf1", n_sample, sample_ns, baseline_wf1);
  TGraph* wf2_gr = plot_waveform<UShort_t>(wf2_br, 0, "wf2", n_sample, sample_ns, baseline_wf2);

  // compute g2 on raw waveforms
  cout<<"compute g2 from baseline-substracted waveforms with batch_size="<<batch_size<<endl;
  compute_g2<UShort_t, n_sample>(max_delay_sample, wf1_br, wf2_br, baseline_wf1, baseline_wf2, "wf", sample_ns, input_n_waveform, batch_size);

  // Setup session for Twensorflow:
  tensorflow::Session* tf_session=create_tf_session(model, n_cpu_core_to_use);
  if (tf_session == NULL) {
    cout<<"ERROR creating TensorFlow session.";
    return 1;
  }

  // setup new branches to store reconstruction in photo electron
  Float_t pe[num_bins_pe];
  TBranch *pe1_br = waveforms->Branch("pe1", &pe, Form("pe1[%d]/F", num_bins_pe) );
  TBranch *pe2_br = waveforms->Branch("pe2", &pe, Form("pe2[%d]/F", num_bins_pe) );

  // Extract pe
  cout<<"extract photo-electrons from waveforms"<<endl;
  extract_pe(tf_session, wf1_br, pe1_br, baseline_wf1, n_sample, num_bins_pe, input_n_waveform);
  extract_pe(tf_session, wf2_br, pe2_br, baseline_wf2, n_sample, num_bins_pe, input_n_waveform);
  waveforms->Write("", TObject::kOverwrite); //write tree as we added branches with pe
  //close TensorFlow seesion
  close_tf_session(tf_session);

  // create histograms and plot of extracted pe from the 1 waveforms
  TGraph* pe1_gr = plot_waveform<Float_t>(pe1_br, 0, "pe1", num_bins_pe, pe_bin_ns, 0.);
  TGraph* pe2_gr = plot_waveform<Float_t>(pe2_br, 0, "pe2", num_bins_pe, pe_bin_ns, 0.);
  histogram_pe(waveforms, "pe1", 100, 0.);
  histogram_pe(waveforms, "pe2", 100, 0.);

  // Compute g2, adding results to datafile
  cout<<"compute g2 from photo-electrons with batch_size="<<batch_size<<endl;
  compute_g2<Float_t, num_bins_pe>(max_delay_bin, pe1_br, pe2_br, 0, 0, "pe", pe_bin_ns, input_n_waveform, batch_size);

  cout<<"close file"<<endl;
  f->Close();
  cout<<"delete stuff"<<endl;
  delete wf1_gr;
  delete wf2_gr;
  delete pe1_gr;
  delete pe2_gr;
  delete f;
}
