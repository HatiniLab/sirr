/*
 * tttReduceTensorImageFilter.h
 *
 *  Created on: Nov 17, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTREDUCETENSORIMAGEFILTER_H_
#define INCLUDE_TTTREDUCETENSORIMAGEFILTER_H_
#include "itkUnaryFunctorImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
namespace ttt{

template<class TTensor,class TScalar>
class ReduceTensorFunctor{
public:
	ReduceTensorFunctor(){}

  ~ReduceTensorFunctor() {};

  bool operator!=( const ReduceTensorFunctor & other ) const
     {
     return !(*this == other);
     }

   bool operator==( const ReduceTensorFunctor & other ) const
     {
     return true;
     }

   TScalar  operator()(const TTensor & a){

	   TScalar result =0;
	   TScalar  weights[]={1.0,2.0,2.0,1.0,2.0,1.0};
		for(int i=0;i<6;i++){
			result+=weights[i]*a[i];
		}
		return result;
	}
};


template<class TInputImage, class TOutputImage>
class ReduceTensorImageFilter : public itk::UnaryFunctorImageFilter<TInputImage,TOutputImage,ReduceTensorFunctor<typename TInputImage::PixelType,typename TOutputImage::PixelType> >{
public:
	  typedef ReduceTensorImageFilter<TInputImage,TOutputImage>         Self;
		  typedef itk::UnaryFunctorImageFilter<TInputImage,TOutputImage,ReduceTensorFunctor<typename TInputImage::PixelType,typename TOutputImage::PixelType> >  Superclass;
		  typedef itk::SmartPointer< Self >  Pointer;

		  /** Method for creation through the object factory. */
		  itkNewMacro(Self);

		  /** Run-time type information (and related methods). */
		  itkTypeMacro(ReduceTensorImageFilter, ImageToImageFilter);


	protected:
		ReduceTensorImageFilter(){

		}
		~ReduceTensorImageFilter(){

		}
	private:
		ReduceTensorImageFilter(const Self &); //purposely not implemented
		void operator=(const Self &);  //purposely not implemented
};

}


#endif /* INCLUDE_TTTREDUCETENSORIMAGEFILTER_H_ */
