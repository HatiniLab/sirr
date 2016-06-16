/*
 * tttMultiplyImageByTensorImageFilter.h
 *
 *  Created on: Nov 17, 2014
 *      Author: morgan
 */

#ifndef _TTTMULTIPLYIMAGEBYTENSORIMAGEFILTER_H_
#define _TTTMULTIPLYIMAGEBYTENSORIMAGEFILTER_H_
#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkBinaryFunctorImageFilter.h>
namespace ttt{

template<class ScalarType,class TensorType> class MultiplyByTensorFunctor{
public:
	MultiplyByTensorFunctor(){}

  ~MultiplyByTensorFunctor() {};

  bool operator!=( const MultiplyByTensorFunctor & other ) const
     {
     return !(*this == other);
     }

   bool operator==( const MultiplyByTensorFunctor & other ) const
     {
     return true;
     }

	TensorType  operator()(const ScalarType & a,const TensorType & b){
		TensorType result;
		for(int i=0;i<6;i++){
			result[i]=a*b[i];
		}
		return result;
	}


};
template<class TInputImage,class TTensorImage> class MultiplyByTensorImageFilter : public itk::BinaryFunctorImageFilter<TInputImage,TTensorImage,TTensorImage,MultiplyByTensorFunctor<typename TInputImage::PixelType,typename TTensorImage::PixelType> > {
public:
	  typedef MultiplyByTensorImageFilter<TInputImage,TTensorImage>         Self;
	  typedef  itk::BinaryFunctorImageFilter<TInputImage,TTensorImage,TTensorImage,MultiplyByTensorFunctor<typename TInputImage::PixelType,typename TTensorImage::PixelType> >  Superclass;
	  typedef itk::SmartPointer< Self >  Pointer;
/** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Run-time type information (and related methods). */
	  itkTypeMacro(MultiplyByTensorImageFilter, BinaryFunctorImageFilter);

protected:
	  MultiplyByTensorImageFilter(){

	  }
	  ~MultiplyByTensorImageFilter(){

	  }
private:
	  MultiplyByTensorImageFilter(const Self &); //purposely not implemented
	  void operator=(const Self &);  //purposely not implemented

};
}
#if 0
template<class TInputImage,class TTensorImage,class TOutputImage=TTensorImage>
class MultiplyByTensorImageFilter: public itk::ImageToImageFilter<TInputImage,TOutputImage>{

public:
	typedef TInputImage InputImageType;
	typedef TTensorImage TensorImageType;
	typedef TOutputImage OutputImageType;


	/** Standard class typedefs. */
	typedef MultiplyByTensorImageFilter<TInputImage,TTensorImage,TOutputImage>         Self;
	typedef itk::ImageToImageFilter<TInputImage,TOutputImage >  Superclass;
	typedef itk::SmartPointer< Self >  Pointer;

	/** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Run-time type information (and related methods). */
	  itkTypeMacro(MultiplyByTensorImageFilter, ImageToImageFilter);

protected:
	  MultiplyByTensorImageFilter(){

	  }
	  ~MultiplyByTensorImageFilter(){}

	  /** Does the real work. */
	  virtual void GenerateData();
private:
	  MultiplyByTensorImageFilter(const Self &);
};
}


#include "tttMultiplyImageByTensorImageFilter.hxx"

#endif
#endif /* INCLUDE_TTTMULTIPLYIMAGEBYTENSORIMAGEFILTER_H_ */
