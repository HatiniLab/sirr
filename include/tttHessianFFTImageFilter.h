/*
 * tttHessianFFTImageFilter.h
 *
 *  Created on: Oct 23, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTHESSIANFFTIMAGEFILTER_H_
#define INCLUDE_TTTHESSIANFFTIMAGEFILTER_H_
#include <itkImageToImageFilter.h>
using namespace itk;
namespace ttt{

template<class TInputImage,class TOutputImage>

class HessianFFTImageFilter : public itk::ImageToImageFilter<TInputImage,TOutputImage>{

public:
	  /** Standard typedefs. */
	  typedef HessianFFTImageFilter                  Self;
	  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
	  typedef SmartPointer< Self >                                    Pointer;
	  typedef SmartPointer< const Self >                              ConstPointer;
	  /** Method for creation through the object factory. */
	  itkNewMacro(Self);
	  /** Runtime information support. */
	  itkTypeMacro(HessianFFTImageFilter,
			  ImageToImageFilter);

	  virtual void GenerateOutputInformation();
	  virtual void GenerateInputRequestedRegion();
	  virtual void EnlargeOutputRequestedRegion(DataObject *output);

protected:

	  virtual void GenerateData();


	  HessianFFTImageFilter(){

	  }
	  ~HessianFFTImageFilter(){

	  }


private:
	  HessianFFTImageFilter(const Self &);
	  void operator=(const Self &);  //purposely not implemented
};

}


#ifndef ITK_MANUAL_INSTANTIATION
#include "tttHessianFFTImageFilter.hxx"
#endif


#endif /* INCLUDE_TTTHESSIANFFTIMAGEFILTER_H_ */
