
#include <TMath.h>
#include <TTree.h>
#include <TH1.h>

#include "root_datafile.h"


Double_t get_baseline(
  TBranch* branch, const Double_t margin_lsb,  const size_t sample_around, 
  const Long64_t n_waveform_input, const size_t n_sample
) {
  Long64_t n_waveform_branch = branch->GetEntryNumber();
  if (n_waveform_branch < 0) {
    cout<<"ERROR reading the branch in get_baseline()"<<endl;
    return 0;
  }
  size_t n_waveform = size_t(n_waveform_branch);
  if ( (n_waveform_input>0) && (n_waveform_input < n_waveform_branch) ) {
    n_waveform =  size_t(n_waveform_input);
  }
  Double_t wf_min, cut;
  UShort_t wf[n_sample];
  Bool_t is_baseline[n_sample];
  branch->SetAddress(&wf);
  size_t n_baseline_sample = 0;
  Double_t sum_baseline_sample = 0;
  for (size_t i=0; i<n_waveform; i++) {
    branch->GetEntry(i);
    wf_min = TMath::MinElement(n_sample, wf);
    cut = wf_min + margin_lsb;
    for (size_t s=0; s<n_sample; s++) {
      is_baseline[s]=true;
    }
    for (size_t s=0; s<n_sample; s++) {
      if (wf[s] > cut) {
        for (size_t s_dis=size_t(s-sample_around); s_dis<=s+sample_around; s_dis++) {
          if ( (s_dis>=0) && (s_dis < n_sample) ) {
            is_baseline[s_dis]=false;
          }
        }
      }
    }
    for (size_t s=0; s<n_sample; s++) {
      if (is_baseline[s]){
        n_baseline_sample++;
        sum_baseline_sample+=wf[s];
      }
    }
  }
  branch->ResetAddress();
  return sum_baseline_sample/n_baseline_sample;
}


Double_t get_pe_rate_MHz(
  TBranch* branch, const Double_t baseline,  const Long64_t n_waveform_input, 
  const size_t n_sample, const Double_t integral_gain, const Double_t sample_ns
) {
  Long64_t n_waveform_branch = branch->GetEntryNumber();
  if (n_waveform_branch < 0) {
    cout<<"ERROR reading the branch in get_pe_rate_MHz()"<<endl;
    return 0;
  }
  size_t n_waveform = size_t(n_waveform_branch);
  if ( (n_waveform_input>0) && (n_waveform_input < n_waveform_branch) ) {
    n_waveform =  size_t(n_waveform_input);
  }
  UShort_t waveform[n_sample];
  branch->SetAddress(&waveform);
  Double_t integrated_adc = 0.;
  for (size_t i=0; i<n_waveform; i++) {
    branch->GetEntry(i);
    for (UInt_t s=0; s<n_sample; s++) {
      integrated_adc += Double_t(waveform[s]) - baseline;
    }
  }
  Double_t integrated_pe = integrated_adc / integral_gain;
  Double_t pe_rate_GHz = integrated_pe / (n_waveform * n_sample * sample_ns);
  branch->ResetAddress();
  return pe_rate_GHz*1000;
}


TH1F* histogram_waveform(TTree *tree, string branch_name, Double_t baseline) {
  Double_t wf_min = tree->GetMinimum(branch_name.c_str()) - baseline;
  Double_t wf_max = tree->GetMaximum(branch_name.c_str()) - baseline;
  UInt_t n_bin = ((UInt_t) (wf_max - wf_min));
  
  TH1F * wf_hist = new TH1F(
    Form("%s_hist",branch_name.c_str()), 
    Form("ADC histogram for %s", branch_name.c_str()) , n_bin, wf_min, wf_max
  );
  tree->Draw(
    Form("%s-%f>>%s_hist", branch_name.c_str(), baseline, branch_name.c_str()) 
  );
  wf_hist->Write("", TObject::kOverwrite);
  return wf_hist;
}


TH1F* histogram_pe(TTree *tree, string branch_name, UInt_t n_bin, Double_t baseline) {
  Double_t pe_min = tree->GetMinimum(branch_name.c_str()) - baseline;
  Double_t pe_max = tree->GetMaximum(branch_name.c_str()) - baseline;
  TH1F * pe_hist = new TH1F(
    Form("%s_hist",branch_name.c_str()), 
    Form("histogram for reconstructed %s",branch_name.c_str()), 
    n_bin, pe_min, pe_max
  );
  tree->Draw(
    Form("%s-%f>>%s_hist", branch_name.c_str(), baseline, branch_name.c_str())
  );
  pe_hist->Write("", TObject::kOverwrite);
  return pe_hist;
}
