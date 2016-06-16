/*
 * tttDeconvolutionPoissonStage.h
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_H_
#define INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_H_
#include <itkImageToImageFilter.h>
#include <tttPoissonShrinkImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkForwardFFTImageFilter.h>
#include <itkInverseFFTImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <tttPoissonShrinkImageFilter.h>
#include <itkComplexConjugateImageAdaptor.h>
#include <itkExtractImageFilter.h>
namespace ttt {
template<class TComplexImage, class TImage> class DeconvolutionPoissonStageImageFilter: public itk::ImageToImageFilter<
		TComplexImage, TImage> {
public:
	typedef DeconvolutionPoissonStageImageFilter<TComplexImage,TImage> Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::ImageToImageFilter<TComplexImage,TImage> Superclass;

	typedef TComplexImage ComplexImageType;
	typedef TImage ImageType;

	itkNewMacro(Self);
	itkTypeMacro(Self,Superclass);

	itkGetMacro(Lambda,double);
	itkSetMacro(Lambda,double);

	virtual void SetCurrentEstimation(const TComplexImage* image) {
		this->SetNthInput(0, const_cast<TComplexImage*>(image));
	}
	virtual typename TComplexImage::ConstPointer GetCurrentEstimation() {
		return static_cast<const TComplexImage *>(this->itk::ProcessObject::GetInput(
				0));
	}

	virtual void SetInputImage(const TImage*  image) {
		this->SetNthInput(1, const_cast<TImage*>(image));
	}

	virtual typename TImage::ConstPointer GetInputImage() {
		return static_cast<const TImage *>(this->itk::ProcessObject::GetInput(1));
	}

	virtual void SetTransferFunction(const TComplexImage* transfer) {
		this->SetNthInput(2, const_cast<TComplexImage*>(transfer));
	}

	virtual const TComplexImage* GetTransferFunction() {
		return static_cast<const TComplexImage *>(this->itk::ProcessObject::GetInput(
				2));
	}

	virtual void SetLagrangeMultiplier(const TImage* transfer) {
		this->SetNthInput(3, const_cast<TImage*>(transfer));
	}
	virtual const TImage* GetLagrangeMultiplier() {
		return static_cast<const TImage *>(this->itk::ProcessObject::GetInput(3));
	}

	virtual TImage* GetShrinkedImage() {
		return dynamic_cast<TImage *>(this->GetOutput(0));
	}
	virtual TComplexImage* GetConjugatedImage() {
		return dynamic_cast<TComplexImage *>(this->GetOutput(1));
	}
protected:
	DeconvolutionPoissonStageImageFilter();

	itk::DataObject::Pointer MakeOutput(unsigned int idx) {
		itk::DataObject::Pointer output;

		switch (idx) {
		case 0:
			output = (TImage::New()).GetPointer();
			break;
		case 1:
			output = (TComplexImage::New()).GetPointer();
			break;
		default:
			std::cerr << "No output " << idx << std::endl;
			output = NULL;
			break;
		}
		return output.GetPointer();
	}

	virtual void SetShrinkedImage(TImage* shrinked) {
		this->SetNthOutput(0, static_cast<TImage*>(shrinked));
	}

	virtual void SetConjugatedImage(
			TComplexImage * deconvoluted) {
		this->SetNthOutput(1, static_cast<TComplexImage*>(deconvoluted));
	}

	~DeconvolutionPoissonStageImageFilter() {

	}
	virtual void GenerateData();

private:
	typedef itk::MultiplyImageFilter<ComplexImageType> ComplexMultiplyType;
	typedef itk::ForwardFFTImageFilter<ImageType,
			ComplexImageType> FFTFilterType;
	typedef itk::InverseFFTImageFilter<ComplexImageType,
			ImageType> IFFTFilterType;

	typedef itk::AddImageFilter<ImageType> AddType;
	typedef ttt::PoissonShrinkImageFilter<ImageType> PoissonShrinkFilterType;
	typedef itk::SubtractImageFilter<ImageType> SubType;
	typedef itk::MultiplyImageFilter<ComplexImageType,
			itk::ComplexConjugateImageAdaptor<ComplexImageType> > ComplexConjugateMultiplyType;
	typedef itk::ComplexConjugateImageAdaptor<ComplexImageType> ConjugateAdaptor;

	typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractFilterType;

	typename ComplexMultiplyType::Pointer m_ComplexMultiplyFilter1;
	typename IFFTFilterType::Pointer m_IFFTFilter1;
	typename AddType::Pointer m_AddFilter1;
	typename PoissonShrinkFilterType::Pointer m_PoissonShrinkFilter1;

	typename SubType::Pointer m_SubFilter1;
	typename FFTFilterType::Pointer m_FFTFilter2;
	typename ComplexConjugateMultiplyType::Pointer m_ComplexConjugateMultiplyFilter2;

	typename ConjugateAdaptor::Pointer m_ConjugateTransferFunction;

	double m_Lambda;

};
}
#include "tttDeconvolutionPoissonStageImageFilter.hxx"

#endif /* INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_H_ */
