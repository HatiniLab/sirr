/*
 * tttMultiplyImageByConjugateTensorImageFilter.h
 *
 *  Created on: Nov 17, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTMULTIPLYTENSORBYCONJUGATETENSORIMAGEFILTER_H_
#define INCLUDE_TTTMULTIPLYTENSORBYCONJUGATETENSORIMAGEFILTER_H_

#include <itkImageToImageFilter.h>

namespace ttt{

template< class Tensor>
class MultiplyTensorByConjugateTensorFunctor
{
public:
	MultiplyTensorByConjugateTensorFunctor(){}

  ~MultiplyTensorByConjugateTensorFunctor() {};
  bool operator!=( const MultiplyTensorByConjugateTensorFunctor & other ) const
    {
    return !(*this == other);
    }

  bool operator==( const MultiplyTensorByConjugateTensorFunctor & other ) const
    {
    return true;
    }
  inline Tensor operator()( const Tensor & a,const Tensor & b){
	  Tensor result;
	  for(int i=0;i<6;i++){
		  result[i]=a[i] *  std::conj(b[i]);
	  }

	  return result;
  }

};
template<class TTensorImage>
class MultiplyTensorByConjugateTensorImageFilter : public itk::BinaryFunctorImageFilter<TTensorImage,TTensorImage,TTensorImage,MultiplyTensorByConjugateTensorFunctor<typename TTensorImage::PixelType> >{
public:
	  typedef MultiplyTensorByConjugateTensorImageFilter<TTensorImage>         Self;
	  typedef  itk::BinaryFunctorImageFilter<TTensorImage,TTensorImage,TTensorImage,MultiplyTensorByConjugateTensorFunctor<typename TTensorImage::PixelType> >  Superclass;
	  typedef itk::SmartPointer< Self >  Pointer;
/** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Run-time type information (and related methods). */
	  itkTypeMacro(MultiplyTensorByConjugateTensorImageFilter, BinaryFunctorImageFilter);

protected:
	  MultiplyTensorByConjugateTensorImageFilter(){

	  }
	  ~MultiplyTensorByConjugateTensorImageFilter(){

	  }
private:
	  MultiplyTensorByConjugateTensorImageFilter(const Self &); //purposely not implemented
	  void operator=(const Self &);  //purposely not implemented
};

}
#endif /* INCLUDE_TTTMULTIPLYTENSORBYCONJUGATETENSORIMAGEFILTER_H_ */
