#include <getopt.h>
#include <iostream>

#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>

#include "zfits_datafile.h"

#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

void usage(){
  cout<<"usage:"<<endl;
  cout<<"\tconvert_zfits <output_rootfile> <input_zfitsfile>"<<endl;
}


int main(int argc, char** argv) {
  if (argc-optind != 2) {
    cout<<"ERROR: the output root file and an input zfits file must be passed as argument."<<endl;
    usage();
    return -1;
  }
  string input = argv[1];
  if ( !file_exists(input) ) {
    cout<<"could not find input file "<<model<<endl;
    return 1;
  }
  cout<<"input: "<<model<<endl;
  string output = argv[2];
  cout<<"output: "<<output<<endl;

  cout<<"convert wavform to ROOT file"<<endl;
  zfits_to_root(output.c_str(), zfits_files, max_delay_sample, sample_ns);
}
