#ifndef EXTRACT_PE_H
#define EXTRACT_PE_H

#include <string>

namespace tensorflow{ class Session; }
class Tbranch;


tensorflow::Session* create_tf_session(const std::string model, const UInt_t n_cpu=4);

void extract_pe (tensorflow::Session* session, TBranch* wf_br, TBranch* output_pe_br, const Double_t baseline=0, const size_t n_sample=4320, const size_t num_bins_pe=34560, const Long64_t n_waveform_input=-1, const size_t batch_size=10, const Double_t sample_ns=4., const Double_t pe_bin_ns=.5);

void close_tf_session(tensorflow::Session * session);

#endif //EXTRACT_PE_H
