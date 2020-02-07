/*

Usage of Tensorflow C++ API based on https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f

Prerequisite: have C++ API of Tensorflow compiled from sources on the machine
(see https://medium.com/@tomdeore/standalone-c-build-tensorflow-opencv-6dc9d8a1412d)

TENSORFLOW_ROOT="/opt/tensorflow/"
cd $TENSORFLOW_ROOT
bazel build --config=opt --copt=-march=native //tensorflow:libtensorflow_cc.so
bazel build --config=opt --copt=-march=native --config=monolitic //tensorflow:libtensorflow_framework.so

run the following comand too to get the right version of protobuf compiled locally
(see http://www.luohanjie.com/2019-07-17/build-tensorflow-c-static-libraries.html)
source ./tensorflow/tensorflow/contrib/makefile/build_all_linux.sh

Then copy the generated libraries to the lib directory:

cd ~/prog/pe-extractor
mkdir lib
cp ${TENSORFLOW_ROOT}/bazel-bin/tensorflow/libtensorflow_cc.so* lib/
cp ${TENSORFLOW_ROOT}/bazel-bin/tensorflow/libtensorflow_framework.so* lib/
cp ${TENSORFLOW_ROOT}/tensorflow/contrib/makefile/gen/protobuf-host/lib/libprotobuf.so* lib/

Then the includes files:

mkdir -p include/tensorflow
cp -r ${TENSORFLOW_ROOT}/bazel-genfiles/tensorflow  include
cp -r ${TENSORFLOW_ROOT}/tensorflow/cc include/tensorflow
cp -r ${TENSORFLOW_ROOT}/tensorflow/core include/tensorflow
cp -r /usr/local/lib/python3.6/site-packages/tensorflow/include ./

*/

//standard stuff
#include <iostream>
#include <fstream>
#include <vector>
// ROOT stuff
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraphErrors.h>
// TensorFlow stuff
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "extract_pe.h"
#include "root_datafile.h"

using namespace std;
using namespace tensorflow;

Session* create_tf_session(const string model, const UInt_t n_cpu) {
  Session* session;
  SessionOptions options;
  options.config.set_intra_op_parallelism_threads(n_cpu);
  options.config.set_inter_op_parallelism_threads(n_cpu);
  options.config.set_use_per_session_threads(false);  
  Status status = NewSession(options, &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return NULL;
  }
  // Read in the protobuf graph of the model
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), model, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return NULL;
  }
  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return NULL;
  }
  return session;
}


void extract_pe(Session* session, TBranch* wf_br, TBranch* output_pe_br, const Double_t baseline, const size_t n_sample, const size_t num_bins_pe, const Long64_t n_waveform_input, const size_t batch_size, const Double_t sample_ns, const Double_t pe_bin_ns) {
  Long64_t n_waveform_branch = wf_br->GetEntryNumber();
  if (n_waveform_branch < 0) {
    cout<<"error reading branch in extract_pe()"<<endl;
    return;
  }
  size_t n_waveform = size_t(n_waveform_branch);
  if ( (n_waveform_input > 0) && (size_t(n_waveform_input) < n_waveform)) {
    n_waveform = size_t(n_waveform_input);
  }

  //initialize variables
  UShort_t waveform[n_sample];
  float pe[num_bins_pe];
  const string wf_id=wf_br->GetName();
  const string pe_id=output_pe_br->GetName();

  // Setup for reading waveforms
  output_pe_br->SetAddress(&pe);
  wf_br->SetAddress(&waveform);

  // Setup input and output TensorFLow graphs
  Tensor input(DT_FLOAT, TensorShape({batch_size, n_sample}));
  TTypes<float, 2>::Tensor input_mapped = input.tensor<float, 2>();
  std::vector<std::pair<string, tensorflow::Tensor> > inputs = {
    { "reshape_input:0", input },
  };
  std::vector<tensorflow::Tensor> outputs; 

  // Itterate over all waveforms 
  size_t n_wf_batch = 0;
  const size_t print_interval = 500;
  chrono::time_point<chrono::steady_clock> start=chrono::steady_clock::now();
  chrono::time_point<chrono::steady_clock> stop;
  for(size_t wf=0; wf<n_waveform; wf++) {
    if( (wf % print_interval==0) || (wf == n_waveform-1) ) {
      stop = chrono::steady_clock::now();
      Double_t duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
      cout<<"extracting pe from "<<output_pe_br->GetName()<<" "<<wf+1<<"/"<<n_waveform;
      if ( (wf != 0) && (wf != n_waveform-1 ) ) {
          cout<<", "<< Double_t(print_interval)/duration*1000<<" wf/s"<<endl;
      } else {
          cout<<endl;
      }
      start=stop;
    }
    // Read the ADC and use them as input for the CNN
    wf_br->GetEntry(wf);
    for (UInt_t s=0; s<n_sample; s++) {
      input_mapped(n_wf_batch, s) = float(waveform[s])-baseline;
    }
    n_wf_batch += 1;
    if( (n_wf_batch < batch_size) && (wf < n_waveform-1) ) {
      continue;
    }
    // Run the session, evaluating the pe proba from the output of the CNN
    Status status = session->Run(inputs, {"flatten/Reshape:0"}, {}, &outputs);
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return;
    }
    TTypes<float, 2>::Tensor output_mapped = outputs[0].tensor<float, 2>();
    // Fill the branch
    for (unsigned int e=0; e<n_wf_batch; e++) {
      for (UInt_t b=0; b<num_bins_pe; b++) {
        pe[b] = (Double32_t) output_mapped(e, b);
      }
      output_pe_br->Fill();
    }
    n_wf_batch = 0;
  }

  // Save branch with photo-electron to datafile 
  //output_pe_br->ResetAddress();
  //wf_br->ResetAddress();
}

void close_tf_session(Session * session) {
  // Free any resources used by the TensorFLow session
  session->Close();
  delete session;
}
