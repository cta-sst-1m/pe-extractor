#ifndef ROOTDATAFILE_IMPL
#define ROOTDATAFILE_IMPL

//standard stuff
#include <fstream> //ifstream
#include <iostream> //cout
#include <chrono> //chrono::steady_clock::now()
// ROOT stuff
#include <TBranch.h>
#include <TGraph.h>

#include "statistics.h"

using namespace std;

template <class sample_type, const size_t n_sample>
inline void compute_g2(Long_t max_delay_sample, TBranch* br_pix1, TBranch* br_pix2, Double_t baseline_wf1, Double_t baseline_wf2, const char id[], const Double_t sample_ns, const Long64_t n_waveform_input=-1, const size_t batch_size=1) {
  // Initalize variables
  Long64_t n_waveform_branch = br_pix1->GetEntryNumber();
  if (n_waveform_branch < 0) {
    cout<<"ERROR reading the branch in compute_g2()"<<endl;
    return;
  }
  if (br_pix2->GetEntryNumber() != n_waveform_branch) {
    cout<<"ERROR in compute_g2(): the number of waveforms differer between pixels"<<endl;
    return;
  }
  size_t n_waveform = size_t(n_waveform_branch);
  if ( (n_waveform_input > 0) && (n_waveform_input < n_waveform_branch ) ) {
    n_waveform = size_t(n_waveform_input);
  }
  std::vector<Long_t> delays_sample;
  std::vector<Double_t> delays_ns, count_delays, sum_wf1_delays, sum_wf2_delays, 
    sum_wf11_delays, sum_wf12_delays, sum_wf22_delays;
  for(Long_t delay_sample=-max_delay_sample; delay_sample<=max_delay_sample; delay_sample++){
    delays_sample.push_back(delay_sample);
    delays_ns.push_back(delay_sample * sample_ns);
    count_delays.push_back(0);
    sum_wf1_delays.push_back(0);
    sum_wf2_delays.push_back(0);
    sum_wf11_delays.push_back(0);
    sum_wf12_delays.push_back(0);
    sum_wf22_delays.push_back(0);
  }

  //Setup for reading branches
  sample_type wf1[n_sample];
  sample_type wf2[n_sample];
  br_pix1->SetAddress(&wf1);
  br_pix2->SetAddress(&wf2);

  //loop over waveforms
  chrono::time_point<chrono::steady_clock> start=chrono::steady_clock::now();
  chrono::time_point<chrono::steady_clock> stop;
  const size_t print_interval = 500;
  size_t current_batch_entry=0;
  float wf1_batch[batch_size][n_sample];
  float wf2_batch[batch_size][n_sample];
  for(size_t wf=0; wf<size_t(n_waveform); wf++) {
    if ( (wf%print_interval == 0) || (wf == n_waveform -1) ) {
      stop = chrono::steady_clock::now();
      Double_t duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
      cout<<"computing g2 for "<<wf+1<<"/"<<n_waveform<<" waveforms";
      if ( (wf != 0) && (wf != size_t(n_waveform-1) ) )  {
          cout<<", "<< Double_t(print_interval)/duration*1000<<" wf/s with ";
          cout<<delays_sample.size()<< " delays."<<endl;
      } else {
          cout<<endl;
      }
      start=stop;
    }
    br_pix1->GetEntry(wf);
    br_pix2->GetEntry(wf);
    //substract baseline
    for (size_t s=0; s<n_sample; s++) {
      wf1_batch[current_batch_entry][s] = float(wf1[s]) - baseline_wf1;
      wf2_batch[current_batch_entry][s] = float(wf2[s]) - baseline_wf2;
    }
    current_batch_entry++;
    //compute correlations on a batch
    if ( (current_batch_entry==batch_size) || (wf == size_t(n_waveform-1) ) ) {
      add_correlation<float, Double_t, n_sample>(
        wf1_batch, wf2_batch, current_batch_entry, delays_sample, 
        count_delays, sum_wf1_delays, sum_wf2_delays, sum_wf11_delays, 
        sum_wf12_delays, sum_wf22_delays 
      );
      current_batch_entry=0;
    }
  }
  br_pix1->ResetAddress();
  br_pix2->ResetAddress();
  //compute g2
  std::vector<Double_t> g2_11, g2_12, g2_22;
  for(size_t d_i=0; d_i<delays_sample.size(); d_i++){
    g2_11.push_back(
      count_delays[d_i]*sum_wf11_delays[d_i]/(sum_wf1_delays[d_i]*sum_wf1_delays[d_i])
    );
    g2_12.push_back(
      count_delays[d_i]*sum_wf12_delays[d_i]/(sum_wf1_delays[d_i]*sum_wf2_delays[d_i])
    );
    g2_22.push_back(
      count_delays[d_i]*sum_wf22_delays[d_i]/(sum_wf2_delays[d_i]*sum_wf2_delays[d_i])
    );
  }
  cout<<"plot graph for g2"<<endl;
  //plots
  TGraph *n_sample_gr = new TGraph(delays_ns.size(), &delays_ns[0], &count_delays[0]);
  n_sample_gr->SetTitle("count of samples for each delay;delay [ns];number of samples");
  n_sample_gr->SetName((string("count_") + id + "_gr").c_str());
  TGraph *sum1_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf1_delays[0]);
  sum1_gr->SetTitle((string("sum1_") + id + ";delay [ns];sum of waveform 1").c_str());
  sum1_gr->SetName((string("sum1_") + id + "_gr").c_str());
  TGraph *sum2_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf2_delays[0]);
  sum2_gr->SetTitle((string("sum2_") + id + ";delay [ns];sum of waveform 2").c_str());
  sum2_gr->SetName((string("sum2_") + id + "_gr").c_str());
  TGraph *sum11_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf11_delays[0]);
  sum11_gr->SetTitle((string("sum11_") + id + ";delay [ns];sum of squared waveform 1").c_str());
  sum11_gr->SetName((string("sum11_") + id + "_gr").c_str());
  TGraph *sum12_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf12_delays[0]);
  sum12_gr->SetTitle((string("sum12_") + id + ";delay [ns];sum of (waveform 1 * waveform 2)").c_str());
  sum12_gr->SetName((string("sum12_") + id + "_gr").c_str());
  TGraph *sum22_gr = new TGraph(delays_ns.size(), &delays_ns[0], &sum_wf22_delays[0]);
  sum22_gr->SetTitle((string("sum22") + id + ";delay [ns];sum of squared waveform 2").c_str());
  sum22_gr->SetName((string("sum22") + id + "_gr").c_str());
  TGraph *g2_11_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_11[0]);
  g2_11_gr->SetTitle((string("g2_") + id + "_11;delay [ns];g2_11").c_str());
  g2_11_gr->SetName((string("g2_") + id + "_11_gr").c_str());
  TGraph *g2_12_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_12[0]);
  g2_12_gr->SetTitle((string("g2_") + id + "_12;delay [ns];g2_12").c_str());
  g2_12_gr->SetName((string("g2_") + id + "_12_gr").c_str());
  TGraph *g2_22_gr = new TGraph(delays_ns.size(), &delays_ns[0], &g2_22[0]);
  g2_22_gr->SetTitle((string("g2_") + id + "_22;delay [ns];g2_22").c_str());
  g2_22_gr->SetName((string("g2_") + id + "_22_gr").c_str());

  //save to file
  cout<<"save g2 plots to file"<<endl;
  n_sample_gr->Write("", TObject::kOverwrite);
  sum1_gr->Write("", TObject::kOverwrite);
  sum2_gr->Write("", TObject::kOverwrite);
  sum11_gr->Write("", TObject::kOverwrite);
  sum12_gr->Write("", TObject::kOverwrite);
  sum22_gr->Write("", TObject::kOverwrite);
  g2_11_gr->Write("", TObject::kOverwrite);
  g2_12_gr->Write("", TObject::kOverwrite);
  g2_22_gr->Write("", TObject::kOverwrite);
  delete n_sample_gr;
  delete sum1_gr;
  delete sum2_gr;
  delete sum11_gr;
  delete sum12_gr;
  delete sum22_gr;
  delete g2_11_gr;
  delete g2_12_gr;
  delete g2_22_gr;
  cout<<"done"<<endl;
}


template <class sample_type>
inline TGraph* plot_waveform(
  TBranch *branch, const size_t waveform_index, const string id, 
  const size_t n_sample, const Double_t sample_ns, const Double_t baseline
) {
  sample_type waveform[n_sample];
  branch->SetAddress(&waveform);
  branch->GetEntry(waveform_index);
  Double_t wf[n_sample];
  Double_t t_sample[n_sample];
  for (size_t s=0; s<n_sample; s++) {
    wf[s] = Double_t(waveform[s]) - baseline;
    t_sample[s] = sample_ns * s;
  }
  TGraph *wf_gr = new TGraph(n_sample, t_sample, wf);
  wf_gr->SetTitle(
    Form("%s for event %zu;time [ns];ADC value [LSB]", id.c_str(), waveform_index) 
  );
  wf_gr->SetName(Form("%s_evt%zu_gr", id.c_str(), waveform_index) );
  branch->ResetAddress();
  wf_gr->Write("", TObject::kOverwrite);
  return wf_gr;
}

#endif //ROOTDATAFILE_IMPL
