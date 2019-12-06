#include <getopt.h>
#include <iostream>
#include <vector>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>

#include "zfits_datafile.h"
#include "root_datafile.h"

#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

void usage(){
  cout<<"usage:"<<endl;
  cout<<"\tconvert_zfits <output_rootfile> <input_zfits_files> .."<<endl;
}


int main(int argc, char** argv) {
  if (argc < 2) {
    cout<<"ERROR: the output root file and at least one input zfits file must be passed as argument."<<endl;
    usage();
    return -1;
  }
  string output = argv[1];
  cout<<"output: "<<output<<endl;

  vector<string> zfits_files;
  for (int i=2; i< argc; i++) {
      if ( !file_exists(argv[i]) ) {
        cout<<"could not find input file "<<argv[i]<<endl;
        continue;
      }
    zfits_files.push_back(argv[i]);
  }
  if (zfits_files.size() < 1) {
    cout<<"No input files, exitting."<<endl;
    return -1;
  }
  cout<<"converting waveforms to ROOT file.."<<endl;
  zfits_to_root(output.c_str(), zfits_files, 4320);
}
