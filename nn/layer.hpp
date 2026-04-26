#ifndef LAYER_HPP
#define LAYER_HPP

#include"matrix.hpp"
#include"tensor_3d.hpp"
#include<functional>
#include<random>
#include<thread>
#include<algorithm>

namespace LibChestNN{
	template<Element T>struct MLPLayer{
		std::function<Matrix<T>(const Matrix<T>&)>activation;
		std::function<Matrix<T>(const Matrix<T>&)>activation_d;
		size_t in_size;
		size_t out_size;
		Matrix<T>W;
		Matrix<T>b;
		Matrix<T>last_input;
		Matrix<T>z;
		bool sm;

		MLPLayer(){
			in_size=0;
			out_size=0;
			W=Matrix<T>();
			b=Matrix<T>();
			last_input=Matrix<T>();
			z=Matrix<T>();
			sm=false;
		}

		MLPLayer(size_t i,size_t o){
			in_size=i;
			out_size=o;
			W.resize(o,i);
			b.resize(o,1);
			last_input.resize(i,1);
			z.resize(o,1);
			sm=false;
		}

		Matrix<T>forward(const Matrix<T>&input){
			Matrix<T>res(out_size,1);
			last_input=input;
			if(input.h==in_size&&input.l==1)z=((W*input)+b);
			res=activation(z);
			return res;
		}

		Matrix<T>backward(const Matrix<T>&dl_da,const T&step){
			Matrix<T>res;
			Matrix<T>dl_dz=dl_da.hadamard(activation_d(z));
			res=W.transpose()*dl_dz;
			W-=step*(dl_dz*last_input.transpose());
			b-=step*dl_dz;
			return res;
		}

		Matrix<T>backward_dz(const Matrix<T>&dl_dz,const T&step){
			Matrix<T>res=W.transpose()*dl_dz;
			W-=step*(dl_dz*last_input.transpose());
			b-=step*dl_dz;
			return res;
		}
	
		void init(T low=T(-1),T high=T(1)){
			static std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<T>dist(low,high);
			for(size_t i=0;i<out_size;++i){
				for(size_t j=0;j<in_size;++j){
					W(i,j)=dist(rng);
				}
				b(i,0)=dist(rng);
			}
		}

		Matrix<T>saveWeight(){
			return W;
		}

		Matrix<T>saveBias(){
			return b;
		}

		bool loadWeight(const Matrix<T>&W){
			if(W.h==this->W.h&&W.l==this->W.l){
				this->W=W;
				return true;
			}
			else return false;
		}

		bool loadBias(const Matrix<T>&b){
			if(b.h==this->b.h&&b.l==this->b.l){
				this->b=b;
				return true;
			}
			return false;
		}
	};

	template<Element T>struct CNNLayer{
		std::function<Tensor3d<T>(const Tensor3d<T>&)>activation;
		std::function<Tensor3d<T>(const Tensor3d<T>&)>activation_d;
		Tensor4d<T>kernel;
		size_t i_c,i_h,i_l;
		size_t o_c,o_h,o_l;
		size_t stride,padding;
		std::vector<T>b;
		Tensor3d<T>z;
		Tensor3d<T>last_input;

		CNNLayer(){
			i_c=0;
			i_h=0;
			i_l=0;
			o_c=0;
			o_h=0;
			o_l=0;
		}

		CNNLayer(
			size_t i_c,
			size_t i_h,
			size_t i_l,
			size_t o_c,
			size_t o_h,
			size_t o_l,
			size_t s,
			size_t p
		){
			this->i_c=i_c;
			this->i_h=i_h;
			this->i_l=i_l;
			this->o_c=o_c;
			this->o_h=o_h;
			this->o_l=o_l;
			this->stride=s;
			this->padding=p;
			b.resize(o_c);
		}

		void init(size_t c_o,size_t c_i,size_t h,size_t l,T low=T(-1),T high=T(1)){
			static std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<T>dist(low,high);
			kernel.resize(c_o);
			for(size_t i=0;i<c_o;i++){
				kernel[i].resize(c_i,h,l);
				for(size_t j=0;j<kernel[i].v.size();j++)kernel[i].v[j]=dist(rng);
			}
			for(size_t i=0;i<b.size();i++)b[i]=dist(rng);
		}

		Tensor3d<T>forward(const Tensor3d<T>&input){
			Tensor3d<T>res;
			last_input=input;
			if(input.c==i_c&&input.h==i_h&&input.l==i_l){
				z=input.convolution(kernel,stride,padding);
				for(size_t i=0;i<o_c;i++)
					for(size_t x=0;x<o_h;x++)
						for(size_t y=0;y<o_l;y++)
							z(i,x,y)+=b[i];
				res=activation(z);
			}
			return res;
		}

		static void da_for(
			size_t from,
			size_t to,
			size_t i_h,
			size_t i_l,
			size_t o_c,
			size_t o_h,
			size_t o_l,
			Tensor3d<T>&res,
			size_t stride,
			size_t padding,
			const Tensor4d<T>&kernel,
			const Tensor3d<T>&dl_dz
		){
			for(size_t j=from;j<to;j++)
				for(size_t a=0;a<i_h;a++){
					for(size_t b=0;b<i_l;b++){
					res(j,a,b)=T(0);
					for(size_t i=0;i<o_c;i++)
						for(size_t u=0;u<o_h;u++)
							for(size_t v=0;v<o_l;v++)
								if(
									!(
										(long long)a-
										(long long)(stride*u+padding)<
										0||
										(long long)b-
										(long long)(stride*v+padding)<
										0||
										(long long)a-
										(long long)(stride*u+padding)>=
										(long long)kernel[0].h||
										(long long)b-
										(long long)(stride*v+padding)>=
										(long long)kernel[0].l
									)
								)
								res(j,a,b)+=
									dl_dz(i,u,v)*
									kernel[i](
										j,
										a-u*stride+padding,
										b-v*stride+padding
									);
					}
				}
		}

		static void grad_for(
			size_t from,
			size_t to,
			Tensor4d<T>&kernel,
			size_t i_h,
			size_t i_l,
			size_t o_c,
			size_t o_h,
			size_t o_l,
			size_t stride,
			size_t padding,
			const Tensor3d<T>&dl_dz,
			const Tensor3d<T>&last_input,
			const T&step
		){
			for(size_t j=from;j<to;j++)
				for(size_t x=0;x<kernel[0].h;x++)
					for(size_t y=0;y<kernel[0].l;y++){
						for(size_t i=0;i<o_c;i++){
							T grad_k=T(0);
							for(size_t u=0;u<o_h;u++)
								for(size_t v=0;v<o_l;v++){
									if(
										!(
											(long long)(u*stride+x)-
											(long long)padding<
											0||
											(long long)(v*stride+y)-
											(long long)padding<
											0||
											(long long)(u*stride+x)-
											(long long)padding>=
											(long long)i_h||
											(long long)(v*stride+y)-
											(long long)padding>=
											(long long)i_l
										)
									)
									grad_k+=
										dl_dz(i,u,v)*
										last_input(
											j,
											u*stride+x-padding,
											v*stride+y-padding
										);
								}
							kernel[i](j,x,y)-=step*grad_k;
						}
					}
		}

		Tensor3d<T>backward(const Tensor3d<T>&dl_da,const T&step){
			Tensor3d<T>res(i_c,i_h,i_l);

			for(size_t i=0;i<res.v.size();i++)res.v[i]=T(0);
			Tensor3d<T>dl_dz=dl_da.hadamard(activation_d(z));

			if(thread_num==0&&!over_threshold(200000,{o_c,o_h,o_l,i_c,i_h,i_l}))
				for(size_t j=0;j<i_c;j++)
					for(size_t a=0;a<i_h;a++){
						for(size_t b=0;b<i_l;b++){
						res(j,a,b)=T(0);
						for(size_t i=0;i<o_c;i++)
							for(size_t u=0;u<o_h;u++)
								for(size_t v=0;v<o_l;v++)
									if(
										!(
											(long long)a-
											(long long)(stride*u+padding)<
											0||
											(long long)b-
											(long long)(stride*v+padding)<
											0||
											(long long)a-
											(long long)(stride*u+padding)>=
											(long long)kernel[0].h||
											(long long)b-
											(long long)(stride*v+padding)>=
											(long long)kernel[0].l
										)
									)
									res(j,a,b)+=
										dl_dz(i,u,v)*
										kernel[i](
											j,
											a-u*stride+padding,
											b-v*stride+padding
										);
						}
					}
			else{
				std::vector<std::thread>ts;
				size_t i=0;
				size_t real_thread_num=std::min(
					static_cast<size_t>(thread_num),
					i_c
				);
				if(real_thread_num==0)real_thread_num=1;
				size_t l=(i_c+real_thread_num-1)/real_thread_num;
				while(i<i_c){
					if(i+l<i_c)
						ts.push_back(
							std::thread(
								da_for,
								i,
								i+l,
								i_h,
								i_l,
								o_c,
								o_h,
								o_l,
								std::ref(res),
								stride,
								padding,
								std::cref(kernel),
								std::cref(dl_dz)
							)
						);
					else
						ts.push_back(
							std::thread(
								da_for,
								i,
								i_c,
								i_h,
								i_l,
								o_c,
								o_h,
								o_l,
								std::ref(res),
								stride,
								padding,
								std::cref(kernel),
								std::cref(dl_dz)
							)
						);
					i+=l;
				}
				for(size_t i=0;i<ts.size();i++)ts[i].join();
			}

			if(thread_num==0&&!over_threshold(200000,{o_c,o_h,o_l,i_c,i_h,i_l}))
				for(size_t j=0;j<i_c;j++)
					for(size_t x=0;x<kernel[0].h;x++)
						for(size_t y=0;y<kernel[0].l;y++){
							for(size_t i=0;i<o_c;i++){
								T grad_k=T(0);
								for(size_t u=0;u<o_h;u++)
									for(size_t v=0;v<o_l;v++){
										if(
											!(
												(long long)(u*stride+x)-
												(long long)padding<
												0||
												(long long)(v*stride+y)-
												(long long)padding<
												0||
												(long long)(u*stride+x)-
												(long long)padding>=
												(long long)i_h||
												(long long)(v*stride+y)-
												(long long)padding>=
												(long long)i_l
											)
										)
										grad_k+=
											dl_dz(i,u,v)*
											last_input(
												j,
												u*stride+x-padding,
												v*stride+y-padding
											);
									}
								kernel[i](j,x,y)-=step*grad_k;
							}
						}
			else{
				std::vector<std::thread>ts;
				size_t i=0;
				size_t real_thread_num=std::min(
					static_cast<size_t>(thread_num),
					i_c
				);
				if(real_thread_num==0)real_thread_num=1;
				size_t l=(i_c+real_thread_num-1)/real_thread_num;
				while(i<i_c){
					if(i+l<i_c)
						ts.push_back(
							std::thread(
								grad_for,
								i,
								i+l,
								std::ref(kernel),
								i_h,
								i_l,
								o_c,
								o_h,
								o_l,
								stride,
								padding,
								std::cref(dl_dz),
								std::cref(last_input),
								std::cref(step)
							)
						);
					else
						ts.push_back(
							std::thread(
								grad_for,
								i,
								i_c,
								std::ref(kernel),
								i_h,
								i_l,
								o_c,
								o_h,
								o_l,
								stride,
								padding,
								std::cref(dl_dz),
								std::cref(last_input),
								std::cref(step)

							)
						);
					i+=l;
				}
				for(size_t i=0;i<ts.size();i++)ts[i].join();
			}

			std::vector<T>grad_b(o_c,T(0));

			for(size_t i=0;i<o_c;i++)
				for(size_t u=0;u<o_h;u++)
					for(size_t v=0;v<o_l;v++)
						grad_b[i]+=dl_dz(i,u,v);

			for(size_t i=0;i<o_c;i++)b[i]-=step*grad_b[i];

			return res;
		}

		Tensor4d<T>saveKernel(){
			return kernel;
		}

		std::vector<T>saveBias(){
			return b;
		}

		bool loadKernel(const Tensor4d<T>&K){
			if(
				K.size()==kernel.size()&&
				K[0].c==kernel[0].c&&
				K[0].h==kernel[0].h&&
				K[0].l==kernel[0].l
			){
				kernel=K;
				return true;
			}
			else return false;
		}

		bool loadBias(const std::vector<T>&b){
			if(b.size()==this->b.size()){
				this->b=b;
				return true;
			}
			else return false;
		}
	};
}

#endif
