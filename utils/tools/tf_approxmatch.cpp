// pointnet-autoencoder

// Copyright (c) 2018, Geometric Computation Group of Stanford University

// The MIT License (MIT)

// Copyright (c) 2018 Charles R. Qi

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <vector>
#include <math.h>
using namespace tensorflow;
REGISTER_OP("ApproxMatch")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("n: int32")
	.Input("m: int32")
	.Output("match: float32");
REGISTER_OP("MatchCost")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("match: float32")
	.Output("cost: float32");
REGISTER_OP("MatchCostGrad")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("match: float32")
	.Output("grad1: float32")
	.Output("grad2: float32");

void approxmatch_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	for (int i=0;i<b;i++){
		int factorl=std::max(n,m)/n;
		int factorr=std::max(n,m)/m;
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		std::vector<double> weight(n*m);
		for (int j=0;j<n*m;j++)
			match[j]=0;
		for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[l*n+k]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[l*n+k];
				}
				for (int l=0;l<m;l++){
					weight[l*n+k]=weight[l*n+k]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[l*n+k];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[l*n+k]*=ss[l];
					s+=weight[l*n+k];
					ss2[l]+=weight[l*n+k];
				}
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}
void approxmatch_cpu_dyn(int b,int n,int m,const float * xyz1,const float * xyz2,float * match, const int * cn, const int * cm){
	for (int i=0;i<b;i++){
		int factorl=std::max(cn[i],cm[i])/cn[i];
		int factorr=std::max(cn[i],cm[i])/cm[i];
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		std::vector<double> weight(n*m);
		for (int j=0;j<n*m;j++){
			match[j]=0;
			weight[j] = 0;
		}
		for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<cn[i];k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<cm[i];l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[l*n+k]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<cn[i];k++){
				double s=1e-9;
				for (int l=0;l<cm[i];l++){
					s+=weight[l*n+k];
				}
				for (int l=0;l<cm[i];l++){
					weight[l*n+k]=weight[l*n+k]/s*saturatedl[k];
				}
				for (int l=0;l<cm[i];l++)
					ss[l]+=weight[l*n+k];
			}
			for (int l=0;l<cm[i];l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			std::vector<double> ss2(m,0);
			for (int k=0;k<cn[i];k++){
				double s=0;
				for (int l=0;l<cm[i];l++){
					weight[l*n+k]*=ss[l];
					s+=weight[l*n+k];
					ss2[l]+=weight[l*n+k];
				}
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<cm[i];l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}
void matchcost_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int k=0;k<n;k++)
			for (int l=0;l<m;l++){
				float x1=xyz1[k*3+0];
				float y1=xyz1[k*3+1];
				float z1=xyz1[k*3+2];
				float x2=xyz2[l*3+0];
				float y2=xyz2[l*3+1];
				float z2=xyz2[l*3+2];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[l*n+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		cost+=1;
	}
}
void matchcostgrad_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2){
	for (int i=0;i<b;i++){
		for (int k=0;k<n;k++)
			grad1[k*3+0]=0;
		for (int l=0;l<m;l++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float x1=xyz1[k*3+0];
				float y1=xyz1[k*3+1];
				float z1=xyz1[k*3+2];
				float x2=xyz2[l*3+0];
				float y2=xyz2[l*3+1];
				float z2=xyz2[l*3+2];
				float d=std::max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				float dx=match[l*n+k]*((x2-x1)/d);
				float dy=match[l*n+k]*((y2-y1)/d);
				float dz=match[l*n+k]*((z2-z1)/d);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[l*3+0]=sx;
			grad2[l*3+1]=sy;
			grad2[l*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		grad1+=n*3;
		grad2+=m*3;
	}
}
void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp);
void approxmatchLauncherDyn(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp,const int * cn, const int * cm);
void matchcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out);
void matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2);

class ApproxMatchGpuOp: public OpKernel{
	public:
		explicit ApproxMatchGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("ApproxMatch expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,n<=4096,errors::InvalidArgument("ApproxMatch handles at most 4096 dataset points"));

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,m<=1024,errors::InvalidArgument("ApproxMatch handles at most 1024 query points"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			const Tensor& cn_tensor=context->input(2);
			OP_REQUIRES(context,((cn_tensor.dims()==2 && cn_tensor.shape().dim_size(1)==1) || cn_tensor.dims()==1) && cn_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,1) n shape, and batch_size must match"));
			auto cn_flat=cn_tensor.flat<int>();
			const int * cn = &(cn_flat(0));

			const Tensor& cm_tensor=context->input(3);
			OP_REQUIRES(context,((cm_tensor.dims()==2 && cm_tensor.shape().dim_size(1)==1) || cm_tensor.dims()==1) && cm_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,1) m shape, and batch_size must match"));
			auto cm_flat=cm_tensor.flat<int>();
			const int * cm = &(cm_flat(0));

			Tensor * match_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,n},&match_tensor));
			auto match_flat=match_tensor->flat<float>();
			float * match=&(match_flat(0));
			Tensor temp_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,(n+m)*2},&temp_tensor));
			auto temp_flat=temp_tensor.flat<float>();
			float * temp=&(temp_flat(0));

			approxmatchLauncherDyn(b,n,m,xyz1,xyz2,match,temp,cn,cm);
		}
};
REGISTER_KERNEL_BUILDER(Name("ApproxMatch").Device(DEVICE_GPU), ApproxMatchGpuOp);
class ApproxMatchOp: public OpKernel{
	public:
		explicit ApproxMatchOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("ApproxMatch expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,n<=4096,errors::InvalidArgument("ApproxMatch handles at most 4096 dataset points"));

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,m<=1024,errors::InvalidArgument("ApproxMatch handles at most 1024 query points"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			const Tensor& cn_tensor=context->input(2);
			OP_REQUIRES(context,((cn_tensor.dims()==2 && cn_tensor.shape().dim_size(1)==1) || cn_tensor.dims()==1) && cn_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,1) n shape, and batch_size must match"));
			auto cn_flat=cn_tensor.flat<int>();
			const int * cn = &(cn_flat(0));

			const Tensor& cm_tensor=context->input(3);
			OP_REQUIRES(context,((cm_tensor.dims()==2 && cm_tensor.shape().dim_size(1)==1) || cm_tensor.dims()==1) && cm_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ApproxMatch expects (batch_size,1) m shape, and batch_size must match"));
			auto cm_flat=cm_tensor.flat<int>();
			const int * cm = &(cm_flat(0));

			Tensor * match_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,n},&match_tensor));
			auto match_flat=match_tensor->flat<float>();
			float * match=&(match_flat(0));
			approxmatch_cpu_dyn(b,n,m,xyz1,xyz2,match,cn,cm);
		}
};
REGISTER_KERNEL_BUILDER(Name("ApproxMatch").Device(DEVICE_CPU), ApproxMatchOp);
class MatchCostGpuOp: public OpKernel{
	public:
		explicit MatchCostGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MatchCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));

			Tensor * cost_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b},&cost_tensor));
			auto cost_flat=cost_tensor->flat<float>();
			float * cost=&(cost_flat(0));
			matchcostLauncher(b,n,m,xyz1,xyz2,match,cost);
		}
};
REGISTER_KERNEL_BUILDER(Name("MatchCost").Device(DEVICE_GPU), MatchCostGpuOp);
class MatchCostOp: public OpKernel{
	public:
		explicit MatchCostOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MatchCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));

			Tensor * cost_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b},&cost_tensor));
			auto cost_flat=cost_tensor->flat<float>();
			float * cost=&(cost_flat(0));
			matchcost_cpu(b,n,m,xyz1,xyz2,match,cost);
		}
};
REGISTER_KERNEL_BUILDER(Name("MatchCost").Device(DEVICE_CPU), MatchCostOp);

class MatchCostGradGpuOp: public OpKernel{
	public:
		explicit MatchCostGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MatchCostGrad expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MatchCostGrad expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MatchCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));

			Tensor * grad1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad1_tensor));
			auto grad1_flat=grad1_tensor->flat<float>();
			float * grad1=&(grad1_flat(0));
			Tensor * grad2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&grad2_tensor));
			auto grad2_flat=grad2_tensor->flat<float>();
			float * grad2=&(grad2_flat(0));
			matchcostgradLauncher(b,n,m,xyz1,xyz2,match,grad1,grad2);
		}
};
REGISTER_KERNEL_BUILDER(Name("MatchCostGrad").Device(DEVICE_GPU), MatchCostGradGpuOp);
class MatchCostGradOp: public OpKernel{
	public:
		explicit MatchCostGradOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MatchCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MatchCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));

			Tensor * grad1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad1_tensor));
			auto grad1_flat=grad1_tensor->flat<float>();
			float * grad1=&(grad1_flat(0));
			Tensor * grad2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&grad2_tensor));
			auto grad2_flat=grad2_tensor->flat<float>();
			float * grad2=&(grad2_flat(0));
			matchcostgrad_cpu(b,n,m,xyz1,xyz2,match,grad1,grad2);
		}
};
REGISTER_KERNEL_BUILDER(Name("MatchCostGrad").Device(DEVICE_CPU), MatchCostGradOp);
