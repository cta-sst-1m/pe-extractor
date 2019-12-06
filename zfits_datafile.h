#ifndef ZFITSDATAFILE_H
#define ZFITSDATAFILE_H

#include <vector>
#include <string>

void zfits_to_root(
  const char output_filename[], const std::vector<std::string> zfits_files,
  const size_t n_sample = 4320, const long long n_waveform=-1
);

#endif //ZFITSDATAFILE_H
