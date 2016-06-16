/*
 * tttHessianShrink.h
 *
 *  Created on: Nov 19, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTHESSIANSHRINKIMAGEFILTER_H_
#define INCLUDE_TTTHESSIANSHRINKIMAGEFILTER_H_
#include <itkUnaryFunctorImageFilter.h>
#include <itkMacro.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
namespace ttt{
template< class TensorType>
class HessianShrinkFunctor{
public:
	typedef typename TensorType::ComponentType ValueType;
	itkStaticConstMacro(Dimension, unsigned int, TensorType::Dimension);

	HessianShrinkFunctor(){
		m_Lambda=itk::NumericTraits<ValueType>::min();
	}
	bool operator!=( const HessianShrinkFunctor & other ) const
		{
	    return !(*this == other);
	    }
	bool operator==( const HessianShrinkFunctor & other ) const
		{
	    return true;
	    }
#if 0
	TensorType operator()(const TensorType  &  tensor)
	{
		vnl_matrix_fixed<ValueType,Dimension,Dimension> tensormatrix;
		for(int r=0;r<Dimension;r++){
			for(int c=0;c<Dimension;c++){
				tensormatrix(r,c)=tensor(r,c);
			}
		}
		vnl_svd<ValueType> calculator(tensormatrix);

		for(int i=0;i<Dimension;i++){
			if(calculator.W(i) < - m_Lambda){
				calculator.W(i)=calculator.W(i)+m_Lambda;
			}else if(calculator.W(i)>m_Lambda){
				calculator.W(i)=calculator.W(i)-m_Lambda;
			}else {
				calculator.W(i)=0;
			}
		}
		vnl_matrix<ValueType> reconstruct =calculator.recompose(Dimension);
		TensorType result;

		for(int r=0;r<Dimension;r++){
			for(int c=0;c<Dimension;c++){
				result(r,c)=reconstruct(r,c);
			}
		}
		//std::cout << tensor <<  " SHR: " << result;
		return result;


	}
#endif
	TensorType operator()(const TensorType  &  tensor)
	{
		vnl_matrix_fixed<ValueType,Dimension,Dimension> tensormatrix;
		for(int r=0;r<Dimension;r++){
			for(int c=0;c<Dimension;c++){
				tensormatrix(r,c)=tensor(r,c);
			}
		}
	 	vnl_symmetric_eigensystem<ValueType> calculator(tensormatrix);

	 	int max=-1;
	 	ValueType value =0;

	 	for(int i=0;i<Dimension;i++){
	 		if(vnl_math_abs(calculator.D(i))>=value){
	 			value=vnl_math_abs(calculator.D(i));
	 			max=i;
	 		}
	 	}
	 	assert(max!=-1);
	 	if(calculator.D(max)>0){
	 		TensorType result;
	 		result.Fill(0.0);
	 		return result;
	 	}else{
			for(int i=0;i<Dimension;i++){
				if(calculator.D(i) < - m_Lambda){
					calculator.D(i)=calculator.D(i)+m_Lambda;
				}else if(calculator.D(i)>m_Lambda){
					calculator.D(i)=calculator.D(i)-m_Lambda;
				}else {
					calculator.D(i)=0;
				}
			}
			vnl_matrix<ValueType> reconstruct =calculator.recompose();
			TensorType result;

			for(int r=0;r<Dimension;r++){
				for(int c=0;c<Dimension;c++){
					result(r,c)=reconstruct(r,c);
				}
			}
			//std::cout << tensor <<  " SHR: " << result;
			return result;
	 	}

	}
	typename TensorType::ComponentType m_Lambda;
};


template<class THessianImage>
class HessianShrinkImageFilter : public itk::UnaryFunctorImageFilter<THessianImage,THessianImage,HessianShrinkFunctor<typename THessianImage::PixelType> >{
public:
	/** Standard class typedefs. */
  typedef HessianShrinkImageFilter<THessianImage>         Self;
  typedef itk::UnaryFunctorImageFilter<THessianImage,THessianImage,HessianShrinkFunctor<typename THessianImage::PixelType> > Superclass;
  typedef itk::SmartPointer< Self >  Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HessianShrinkImageFilter, UnaryFunctorImageFilter);
protected:
  HessianShrinkImageFilter(){}
  ~HessianShrinkImageFilter(){}


private:
  HessianShrinkImageFilter(const Self &); //purposely not implemented
	void operator=(const Self &);  //purposely not implemented
};
}


#endif /* INCLUDE_TTTHESSIANSHRINKIMAGEFILTER_H_ */
