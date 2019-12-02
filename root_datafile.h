#ifndef ROOT_DATAFILE_H
#define ROOT_DATAFILE_H

#include "root_datafile-impl.cpp"

class TH1I;
class TH1F;
class TTree;
class TBranch;

Double_t get_baseline(
  TBranch* branch, const Double_t margin_lsb=8., 
  const size_t sample_around=20, const Long64_t n_waveform=1000, 
  const size_t n_sample=4320
);

Double_t get_pe_rate_MHz(
  TBranch* branch, const Double_t baseline, 
  const Long64_t n_waveform=1000, const size_t n_sample=4320, 
  const Double_t integral_gain=21., const Double_t sample_ns=4.
);

inline bool file_exists (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
};

TH1F* histogram_waveform(TTree *tree, string branch_name, Double_t baseline=0.);

TH1F* histogram_pe(TTree *tree, string branch_name, UInt_t n_bin=100, Double_t baseline=0.);
#endif //ROOT_DATAFILE_H
