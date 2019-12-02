#ifndef STATISCS_IMPL
#define STATISCS_IMPL

#include <vector>


template <class sample_type, class sum_type, const size_t n_sample>
inline void add_correlation(
    sample_type samples1[][n_sample], sample_type samples2[][n_sample],
    size_t batch_size,
    const std::vector<Long_t> delays_sample,
    std::vector<sum_type> &count_delays,
    std::vector<sum_type> &sum_sample1_delays,
    std::vector<sum_type> &sum_sample2_delays,
    std::vector<sum_type> &sum_sample11_delays,
    std::vector<sum_type> &sum_sample12_delays,
    std::vector<sum_type> &sum_sample22_delays 
){
  for(size_t d_i=0; d_i < delays_sample.size(); d_i++){
    Long_t delay_sample = delays_sample[d_i];
    if(delay_sample<0){
      for(size_t b=0;b<batch_size;b++) {
        for(size_t index_sample1=0; index_sample1<(n_sample+delay_sample); index_sample1++){
          long index_sample2=index_sample1 - delay_sample;
          count_delays[d_i]+=1;
          double sample1=samples1[b][index_sample1];
          double sample2=samples2[b][index_sample2];
          sum_sample1_delays[d_i]+=sample1;
          sum_sample2_delays[d_i]+=sample2;
          sum_sample11_delays[d_i]+=sample1*sample1;
          sum_sample12_delays[d_i]+=sample1*sample2;
          sum_sample22_delays[d_i]+=sample2*sample2;
        }
      }
    }
    if(delay_sample>=0){
      for(size_t b=0; b<batch_size; b++) {
        for(size_t index_sample2=0; index_sample2<(n_sample-delay_sample); index_sample2++){
          size_t index_sample1=index_sample2 + delay_sample;
          count_delays[d_i]+=1;
          double sample1=samples1[b][index_sample1];
          double sample2=samples2[b][index_sample2];
          sum_sample1_delays[d_i]+=sample1;
          sum_sample2_delays[d_i]+=sample2;
          sum_sample11_delays[d_i]+=sample1*sample1;
          sum_sample12_delays[d_i]+=sample1*sample2;
          sum_sample22_delays[d_i]+=sample2*sample2;
        }
      }
    }
  }
}

#endif //STATISCS_IMPL
