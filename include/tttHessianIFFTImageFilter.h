/*
 * tttHessianIFFTImageFilter.h
 *
 *  Created on: Oct 23, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTHESSIANIFFTIMAGEFILTER_H_
#define INCLUDE_TTTHESSIANIFFTIMAGEFILTER_H_
#include <itkImageToImageFilter.h>
using namespace itk;
namespace ttt{

template<class TInputImage,class TOutputImage>

class HessianIFFTImageFilter : public itk::ImageToImageFilter<TInputImage,TOutputImage>{

public:
	  /** Standard typedefs. */
	  typedef HessianIFFTImageFilter                  Self;
	  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
	  typedef SmartPointer< Self >                                    Pointer;
	  typedef SmartPointer< const Self >                              ConstPointer;
	  /** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Runtime information support. */
	  itkTypeMacro(HessianIFFTImageFilter,
			  ImageToImageFilter);

protected:
	  virtual void GenerateData();

	  HessianIFFTImageFilter(){

	  }
	  ~HessianIFFTImageFilter(){

	  }

	  /** The output may be a different size from the input if complex conjugate
	 * symmetry is implicit. */
	 virtual void GenerateOutputInformation();
	 /** This class requires the entire input. */
	 virtual void GenerateInputRequestedRegion();
	 /** Sets the output requested region to the largest possible output
	 * region. */
	 void EnlargeOutputRequestedRegion( DataObject *itkNotUsed(output) );
private:
	  HessianIFFTImageFilter(const Self &);
	  void operator=(const Self &);  //purposely not implemented
};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "tttHessianIFFTImageFilter.hxx"
#endif

#endif /* INCLUDE_TTTHESSIANIFFTIMAGEFILTER_H_ */
