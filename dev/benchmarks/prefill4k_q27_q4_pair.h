#pragma once
// Private two-group scheduling alternative. Each original K64 affine epilogue
// executes in group order. Partial tensors are never added before scaling.
template<ushort M, ushort N, ushort SG>
inline void prefill4k_q27_pair_tile(device bfloat *input,device uchar *weights,
    device bfloat *scales,device bfloat *biases,device bfloat *output,
    device const float *sums,uint output_size,uint input_size,uint output_origin) {
  constexpr ushort StorageN=256;
  const uint groups=input_size/64,tile=output_origin/StorageN,column=output_origin%StorageN;
  auto a=tensor(input,dextents<int,2>{int(input_size),M},array<int,2>{1,int(input_size)});
  device uchar *tile_weights=weights+(ulong(tile)*groups*StorageN+column)*32;
  tensor<device uint4b_format,dextents<int,2>,tensor_inline> first_b(
      tile_weights,dextents<int,2>{64,N},array<int,2>{1,64});
  auto first_a=a.slice<64,M>(0,0);
  auto first_slice_b=first_b.slice<64,N>(0,0);
  constexpr auto descriptor=matmul2d_descriptor(M,N,64,false,true,false);
  matmul2d<descriptor,execution_simdgroups<SG>> operation;
  auto accumulated=operation.template get_destination_cooperative_tensor<
      decltype(first_a),decltype(first_slice_b),float>();
  const bool fullyOccupied=uint(accumulated.get_capacity())*uint(SG)*32u==uint(M)*N;
  const auto traversal=fullyOccupied?Q4Traversal::All:q4_traversal(accumulated);
  q4_visit(accumulated,traversal,[&](ushort i){accumulated[i]=0.0f;});
  for(uint group=0;group<groups;group+=2) {
    auto a0=a.slice<64,M>(group*64,0);
    tensor<device uint4b_format,dextents<int,2>,tensor_inline> b0(
        tile_weights+ulong(group)*StorageN*32,dextents<int,2>{64,N},array<int,2>{1,64});
    auto slice_b0=b0.slice<64,N>(0,0);
    auto partial0=operation.template get_destination_cooperative_tensor<
        decltype(a0),decltype(slice_b0),float>();
    operation.run(a0,slice_b0,partial0);
    if(group+1<groups) {
      auto a1=a.slice<64,M>((group+1)*64,0);
      tensor<device uint4b_format,dextents<int,2>,tensor_inline> b1(
          tile_weights+ulong(group+1)*StorageN*32,dextents<int,2>{64,N},array<int,2>{1,64});
      auto slice_b1=b1.slice<64,N>(0,0);
      auto partial1=operation.template get_destination_cooperative_tensor<
          decltype(a1),decltype(slice_b1),float>();
      operation.run(a1,slice_b1,partial1);
      q4_visit(accumulated,traversal,[&](ushort i) {
        const auto index=accumulated.get_multidimensional_index(i);
        const uint row=index[1];
        const ulong parameter=(ulong(tile)*groups+group)*StorageN+column+index[0];
        accumulated[i]+=partial0[i]*float(scales[parameter])+sums[group*M+row]*float(biases[parameter]);
        accumulated[i]+=partial1[i]*float(scales[parameter+StorageN])+sums[(group+1)*M+row]*float(biases[parameter+StorageN]);
      });
    }else {
      q4_visit(accumulated,traversal,[&](ushort i) {
        const auto index=accumulated.get_multidimensional_index(i);
        const ulong parameter=(ulong(tile)*groups+group)*StorageN+column+index[0];
        accumulated[i]+=partial0[i]*float(scales[parameter])+sums[group*M+index[1]]*float(biases[parameter]);
      });
    }
  }
  q4_visit(accumulated,traversal,[&](ushort i) {
    const auto index=accumulated.get_multidimensional_index(i);
    const float value=float(bfloat(accumulated[i]));
    output[ulong(index[1])*output_size+output_origin+index[0]]=bfloat(value);
  });
}
