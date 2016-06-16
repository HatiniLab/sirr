/*
 * outlinemfdeconvolution.cpp
 *
 *  Created on: Nov 20, 2014
 *      Author: morgan
 */
//#include <tttMultiframeHessianRegularizedDeconvolutionImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVariationalRegistrationMultiResolutionFilter.h>
#include <itkVariationalSymmetricDiffeomorphicRegistrationFilter.h>
#include <itkVariationalRegistrationStopCriterion.h>
#include <itkVariationalRegistrationLogger.h>
//#include <itkVariationalRegistrationNCCFunction.h>
#include <itkVariationalRegistrationSSDFunction.h>
#include <itkVariationalRegistrationDemonsFunction.h>
#include <itkVariationalRegistrationElasticRegularizer.h>
#include <itkVariationalRegistrationGaussianRegularizer.h>

#include <tttDeconvolutionPoissonStageImageFilter.h>
#include <tttDeconvolutionHessianStageImageFilter.h>
#include <tttDeconvolutionBoundsStageImageFilter.h>
#include <itkPadImageFilter.h>

#include <itkNormalizeToConstantImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkCyclicShiftImageFilter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkSCIFIOImageIO.h>
//#include <itkContinuousBorderWarpImageFilter.h>
#include <tttCentralDifferenceHessianSource.h>
//#include "tttGaussianHessianSource.h"
#include <tttTensorToEnergyImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkComplexToModulusImageAdaptor.h>
#include <tttBoundsShrinkImageFilter.h>

#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#include <itkHalfHermitianToRealInverseFFTImageFilter.h>
#include <itkRecursiveMultiResolutionPyramidImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkMaximumImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkNearestNeighborExtrapolateImageFunction.h>
#include "TrackingAndDeconvolutionProject.h"
#include <itkVideoToVideoFilter.h>
#include <itkImageFilterToVideoFilterWrapper.h>
#include <itkLaplacianImageFilter.h>
#include <itkTernaryFunctorImageFilter.h>

#include <itkNumericTraits.h>
#include <itkMultiplyImageFilter.h>
#include <itkComplexConjugateImageAdaptor.h>
#include <itkPadImageFilter.h>
//#include <itkDemonsRegistrationFilter.h>
//#include <itkLevelSetMotionRegistrationFilter.h>
//#include <itkVideoFileReader.h>
//#include <itkVideoFileWriter.h>
//#include <itkVideoIOBase.h>
//#include <ITKVideoIOExport.h>
//#include "MyDirectoryVideoIO.h"


//#define ALPHAPSF pow(2.0,-20)
//#define ALPHAPSF pow(2.0,-10)
#define ALPHAPSF pow(2.0,2)
#define ALPHAHESSIAN pow(2.0,2)


template<class TReal>
class L1L2Regularization {
public:
	L1L2Regularization() {
		m_Lambda = itk::NumericTraits<TReal>::min();
	}
	;
	~L1L2Regularization() {
	}
	;
	bool operator!=(const L1L2Regularization & other) const {
		return !(*this == other);
	}
	bool operator==(const L1L2Regularization & other) const {
		return true;
	}
	inline TReal operator()(const TReal & e, const TReal & x)

	{
		//return std::max( e * x / ( 1 + (x )*m_Lambda   ), NumericTraits< TReal >::Zero );
		//return std::max(e * x / (1 + x * m_Lambda), NumericTraits<TReal>::Zero);
		//return std::max(e * x / (1 - x * m_Lambda), NumericTraits<TReal>::Zero);
		//return std::max( e * x , NumericTraits< TReal >::Zero );
		return std::max( e * x / ( 1 + m_Lambda   ), itk::NumericTraits< TReal >::Zero );
	}
	TReal m_Lambda;
};


template<class TReal>
class TVL1Regularization {
public:
	TVL1Regularization() {
		m_Lambda = itk::NumericTraits<TReal>::min();
	}
	;
	~TVL1Regularization() {
	}
	;
	bool operator!=(const TVL1Regularization & other) const {
		return !(*this == other);
	}
	bool operator==(const TVL1Regularization & other) const {
		return true;
	}
	inline TReal operator()(const TReal & e, const TReal & x,const TReal & l)

	{
		//return std::max( e * x / ( 1 + (x )*m_Lambda   ), NumericTraits< TReal >::Zero );
		//return std::max(e * x / (1 + x * m_Lambda), NumericTraits<TReal>::Zero);
		return std::max(e * x / (1 - l * m_Lambda), itk::NumericTraits<TReal>::Zero);
		//return std::max( e * x , NumericTraits< TReal >::Zero );
		//return std::max( e * x / ( 1 + m_Lambda   ), NumericTraits< TReal >::Zero );
	}
	TReal m_Lambda;
};

template<class TImage, class TComplexImage> void BlindRL(const typename TComplexImage::Pointer & currentTransferEstimate,
		const typename TComplexImage::Pointer & currentImageEstimate, const typename TImage::Pointer & paddedOriginal, typename TImage::Pointer & psf,
		typename TComplexImage::Pointer & transferNext) {

	typedef itk::MultiplyImageFilter<TComplexImage, TComplexImage, TComplexImage> ComplexMultiplyType;
	// Set up minipipeline to compute estimate at each iteration
	typename ComplexMultiplyType::Pointer complexMultiplyFilter1 = ComplexMultiplyType::New();

	// Transformed estimate will be set as input 1 in Iteration()
	complexMultiplyFilter1->SetInput1(currentImageEstimate);
	complexMultiplyFilter1->SetInput2(currentTransferEstimate);

	typedef itk::HalfHermitianToRealInverseFFTImageFilter<TComplexImage, TImage> IFFTFilterType;
	typename IFFTFilterType::Pointer iFFTFilter1 = IFFTFilterType::New();
	iFFTFilter1->SetInput(complexMultiplyFilter1->GetOutput());

	typedef itk::DivideImageFilter<TImage, TImage, TImage> DivideFilterType;
	typename DivideFilterType::Pointer divideFilter = DivideFilterType::New();

	divideFilter->SetInput1(paddedOriginal);
	divideFilter->SetInput2(iFFTFilter1->GetOutput());

	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TImage, TComplexImage> FFTFilterType;

	typename FFTFilterType::Pointer fFTFilter = FFTFilterType::New();

	fFTFilter->SetInput(divideFilter->GetOutput());

	typedef itk::ComplexConjugateImageAdaptor<TComplexImage> ConjugateAdaptorType;
	typename ConjugateAdaptorType::Pointer conjugateAdaptor = ConjugateAdaptorType::New();
	conjugateAdaptor->SetImage(currentImageEstimate);

	typedef itk::MultiplyImageFilter<TComplexImage, ConjugateAdaptorType, TComplexImage> ComplexConjugateMultiplyType;
	typename ComplexConjugateMultiplyType::Pointer complexMultiplyFilter2 = ComplexConjugateMultiplyType::New();

	complexMultiplyFilter2->SetInput1(fFTFilter->GetOutput());
	complexMultiplyFilter2->SetInput2(conjugateAdaptor);

	typename IFFTFilterType::Pointer iFFTFilter2 = IFFTFilterType::New();

	iFFTFilter2->SetInput(complexMultiplyFilter2->GetOutput());

	typename IFFTFilterType::Pointer iFFTFilter3 = IFFTFilterType::New();

	iFFTFilter3->SetInput(currentTransferEstimate);
#if 0
	typedef itk::MultiplyImageFilter<TImage, TImage, TImage> MultiplyType;

	typename MultiplyType::Pointer multiply = MultiplyType::New();

	multiply->SetInput1(iFFTFilter2->GetOutput());

	multiply->SetInput2(iFFTFilter3->GetOutput());
#endif

	typedef itk::LaplacianImageFilter<TImage,TImage> LaplacianFilterType;
	typename LaplacianFilterType::Pointer laplacianFilter= LaplacianFilterType::New();
	laplacianFilter->SetInput(iFFTFilter3->GetOutput());
	// multiply the result with the input
	typedef itk::TernaryFunctorImageFilter<TImage, TImage, TImage,TImage, TVL1Regularization<double> > RegularizerFilterType;

	typename RegularizerFilterType::Pointer regularizer = RegularizerFilterType::New();
	regularizer->SetInput(0, iFFTFilter2->GetOutput());
	regularizer->SetInput(1, iFFTFilter3->GetOutput());
	regularizer->SetInput(2, laplacianFilter->GetOutput());
	regularizer->GetFunctor().m_Lambda = ALPHAPSF;
	regularizer->Update();

	typedef itk::NormalizeToConstantImageFilter<TImage, TImage> NormalizeFilterType;
	typename NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
	normalizeFilter->SetConstant(itk::NumericTraits<double>::OneValue());

	normalizeFilter->SetInput(regularizer->GetOutput());
	normalizeFilter->Update();

	psf = normalizeFilter->GetOutput();
	psf->DisconnectPipeline();
#if 0
	typedef itk::MaximumImageFilter<TImage, TImage, TImage> MaximumFilterType;

	typename MaximumFilterType::Pointer maximumFilter = MaximumFilterType::New();
	maximumFilter->SetInput1(multiply->GetOutput());
	maximumFilter->SetConstant2(0.0);

	psf = normalizeFilter->GetOutput();
	psf->DisconnectPipeline();
#endif
	typename FFTFilterType::Pointer fFTFilter2 = FFTFilterType::New();

	fFTFilter2->SetInput(normalizeFilter->GetOutput());

	fFTFilter2->Update();

	transferNext = fFTFilter2->GetOutput();
	transferNext->DisconnectPipeline();

}

template<class TensorType> class PlatenessFunctor {
public:
	typedef typename TensorType::ComponentType ValueType;itkStaticConstMacro(Dimension, unsigned int, TensorType::Dimension);

	PlatenessFunctor() {
		m_Alpha=0.5;
		m_Beta=0.5;
		m_Gamma=8;
		m_C = 10e-6;
	}
	bool operator!=( const PlatenessFunctor & other ) const
	{
		return !(*this == other);
	}
	bool operator==( const PlatenessFunctor & other ) const
	{
		return true;
	}
	ValueType operator()(const TensorType & tensor) {

		typename TensorType::EigenValuesArrayType eigenValue;

		tensor.ComputeEigenValues(eigenValue);
		//

		double result = 0;
		// Find the smallest eigenvalue
		double smallest = vnl_math_abs(eigenValue[0]);
		double Lambda1 = eigenValue[0];
		for (unsigned int i = 1; i <= 2; i++) {
			if (vnl_math_abs(eigenValue[i]) < smallest) {
				Lambda1 = eigenValue[i];
				smallest = vnl_math_abs(eigenValue[i]);
			}
		}

		// Find the largest eigenvalue
		double largest = vnl_math_abs(eigenValue[0]);
		double Lambda3 = eigenValue[0];

		for (unsigned int i = 1; i <= 2; i++) {
			if (vnl_math_abs(eigenValue[i] > largest)) {
				Lambda3 = eigenValue[i];
				largest = vnl_math_abs(eigenValue[i]);
			}
		}

		//  find Lambda2 so that |Lambda1| < |Lambda2| < |Lambda3|
		double Lambda2 = eigenValue[0];

		for (unsigned int i = 0; i <= 2; i++) {
			if (eigenValue[i] != Lambda1 && eigenValue[i] != Lambda3) {
				Lambda2 = eigenValue[i];
				break;
			}

		}
		if(Lambda3>0) {
			return 0;
		} else {
			//	return vnl_math_abs(Lambda1) + vnl_math_abs(Lambda2) + vnl_math_abs(Lambda3);
			return std::sqrt(Lambda1*Lambda1 +Lambda2*Lambda2 + Lambda3*Lambda3);
		}
#if 0
		if (Lambda3 >= 0.0 || vnl_math_abs(Lambda3) < 1.0e-6) {
			return 0;
		} else {

			double Lambda1Abs = vnl_math_abs(Lambda1);
			double Lambda2Abs = vnl_math_abs(Lambda2);
			double Lambda3Abs = vnl_math_abs(Lambda3);

			double Lambda1Sqr = vnl_math_sqr(Lambda1);
			double Lambda2Sqr = vnl_math_sqr(Lambda2);
			double Lambda3Sqr = vnl_math_sqr(Lambda3);

			double AlphaSqr = vnl_math_sqr(m_Alpha);
			double BetaSqr = vnl_math_sqr(m_Beta);
			double GammaSqr = vnl_math_sqr(m_Gamma);
			double A {Lambda2Abs / Lambda3Abs};
			double B = vcl_sqrt(vnl_math_abs(Lambda1 * Lambda2)) / (Lambda3Abs); //vessel vs plate vs ball
			double S = vcl_sqrt(Lambda1Sqr + Lambda2Sqr + Lambda3Sqr);

			double vesMeasure_1 = (vcl_exp(
							-0.5 * ((vnl_math_sqr(A)) / (AlphaSqr))));

			double vesMeasure_2 = vcl_exp(
					-0.5 * ((vnl_math_sqr(B)) / (BetaSqr)));

			double vesMeasure_3 = (1
					- vcl_exp(-1.0 * ((vnl_math_sqr(S)) / (2.0 * (GammaSqr)))));

			double vesMeasure_4 = vcl_exp(
					-1.0 * (2.0 * vnl_math_sqr(m_C)) / (Lambda3Sqr));

			double vesselnessMeasure = vesMeasure_1 * vesMeasure_2
			* vesMeasure_3 * vesMeasure_4;
			result=vesselnessMeasure;
#if 0
			if (m_ScalePlatenessMeasure) {
				result = Lambda3Abs * vesselnessMeasure;
			} else {
				result=vesselnessMeasure;
			}
#endif
			return result;

		}
#endif
	}
	typename TensorType::ComponentType m_Alpha;
	typename TensorType::ComponentType m_Beta;
	typename TensorType::ComponentType m_Gamma;
	typename TensorType::ComponentType m_C;
};

template<class TInputImage, class TComplexImage> void ImageFFT(const typename TInputImage::Pointer & input,
		typename TComplexImage::Pointer & transformed) {

	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TInputImage, TComplexImage> FFTFilterType;
	// Take the Fourier transform of the padded image.
	typename FFTFilterType::Pointer imageFFTFilter = FFTFilterType::New();
	//imageFFTFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	imageFFTFilter->SetInput(input);

	imageFFTFilter->Update();

	transformed = imageFFTFilter->GetOutput();
	transformed->DisconnectPipeline();

}
template<class TImage, class TComplexImage> void ImageIFFT(const typename TComplexImage::Pointer & fourier, typename TImage::Pointer & image) {
	//typedef itk::InverseFFTImageFilter<TComplexImage,TImage> IFFTFilterType;
	typedef itk::HalfHermitianToRealInverseFFTImageFilter<TComplexImage, TImage> IFFTFilterType;
	typename IFFTFilterType::Pointer ifft = IFFTFilterType::New();
	ifft->SetInput(fourier);
	ifft->Update();
	image = ifft->GetOutput();
	image->DisconnectPipeline();

}
template<class TInputImage> void PadImage(const typename TInputImage::Pointer & frame, const typename TInputImage::SizeType & padSize,
		const typename TInputImage::SizeType & padLowerBound, typename TInputImage::Pointer & padded) { //const typename TInputImage::Pointer & psf,typename TInputImage::Pointer & padded){
	//typename TInputImage::SizeType padSize = GetPadSize<TInputImage>(frame,psf);
	//typename TInputImage::SizeType padLowerBound = GetPadLowerBound<TInputImage>(frame,psf);

	typename TInputImage::SizeType inputSize = frame->GetLargestPossibleRegion().GetSize();
	typedef itk::PadImageFilter<TInputImage, TInputImage> InputPadFilterType;
	typename InputPadFilterType::Pointer inputPadder = InputPadFilterType::New();
	typedef typename itk::ZeroFluxNeumannBoundaryCondition<TInputImage> BoundaryConditionType;

	BoundaryConditionType * boundaryCondition = new BoundaryConditionType { };
	inputPadder->SetBoundaryCondition(boundaryCondition);

	inputPadder->SetPadLowerBound(padLowerBound);

	typename TInputImage::SizeType inputUpperBound { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		inputUpperBound[i] = (padSize[i] - inputSize[i]) / 2;
		if ((padSize[i] - inputSize[i]) % 2 == 1) {
			inputUpperBound[i]++;
		}
	}
	inputPadder->SetPadUpperBound(inputUpperBound);
	//inputPadder->SetNumberOfThreads( this->GetNumberOfThreads() );
	inputPadder->SetInput(frame);

	inputPadder->Update();
	padded = inputPadder->GetOutput();
	padded->DisconnectPipeline();

}
template<class TImage> typename TImage::Pointer GenerateIdentityPSF(const typename TImage::SizeType size){
	typename TImage::Pointer psf = TImage::New();
	typename TImage::RegionType region;
	typename TImage::IndexType index;
	index.Fill(0);
	region.SetSize(size);
	region.SetIndex(index);
	std::cout << region << std::endl;
	psf->SetRegions(region);
	psf->Allocate();
	psf->FillBuffer(0.0);
	std::cout << "foo" << std::endl;


	typename TImage::IndexType center;

	for(int d=0;d<TImage::ImageDimension;d++){
		center[d]=size[d]/2;
	}


	psf->SetPixel(center,1.0);
	psf->DisconnectPipeline();
	return psf;

}
template<class TInputImage, class TComplexImage>
void DoBoundsStage(const typename TInputImage::Pointer & currentEstimate, const typename TInputImage::Pointer & lagrange,
		const typename TInputImage::SizeType & padSize, const typename TInputImage::SizeType & padLowerBound, typename TInputImage::Pointer & bounded,
		typename TInputImage::Pointer & conjugatedBounded) {

	typedef ttt::DeconvolutionBoundsStageImageFilter<TInputImage, TComplexImage> BoundsStageType;

	typename BoundsStageType::Pointer boundsStage = BoundsStageType::New();

	typedef itk::AddImageFilter<TInputImage, TInputImage, TInputImage> AddType;
	typedef itk::SubtractImageFilter<TInputImage, TInputImage, TInputImage> SubType;

	typename AddType::Pointer addFilter2 = AddType::New();
	addFilter2->SetInput1(currentEstimate);
	addFilter2->SetInput2(lagrange);

	typedef ttt::BoundsShrinkImageFilter<TInputImage> BoundsShrinkFilterType;
	typename BoundsShrinkFilterType::Pointer boundsShrinkFilter1 = BoundsShrinkFilterType::New();

	boundsShrinkFilter1->SetInput(addFilter2->GetOutput());
	boundsShrinkFilter1->Update();
	bounded = boundsShrinkFilter1->GetOutput();
	bounded->DisconnectPipeline();

	typename SubType::Pointer subFilter2 = SubType::New();

	subFilter2->SetInput1(bounded);
	subFilter2->SetInput2(lagrange);

	subFilter2->Update();
	conjugatedBounded = subFilter2->GetOutput();
#if 0
	typename TInputImage::Pointer tmp = subFilter2->GetOutput();
	PadImage<TInputImage>(tmp, padSize, padLowerBound, tmp);

	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TInputImage, TComplexImage> FFTFilterType;

	typename FFTFilterType::Pointer FFTFilter3 = FFTFilterType::New();
	FFTFilter3->SetInput(tmp);

	FFTFilter3->Update();
	conjugatedBounded = FFTFilter3->GetOutput();
#endif
	conjugatedBounded->DisconnectPipeline();
}
template<class TImage> void CropImage(const typename TImage::Pointer & extended, const typename TImage::RegionType & cropRegion,
		typename TImage::Pointer & result) {
	typedef itk::ExtractImageFilter<TImage, TImage> ExtractFilterType;
	typename ExtractFilterType::Pointer extractor = ExtractFilterType::New();
	extractor = ExtractFilterType::New();
	extractor->SetInput(extended);
	extractor->SetExtractionRegion(cropRegion);
	extractor->Update();
	result = extractor->GetOutput();
	result->DisconnectPipeline();

}
template<class TComplexImage, class THessian, class TComplexHessian>
void DoHessianStage(const typename TComplexImage::Pointer & frame, const typename THessian::Pointer & lagrange,
		const typename TComplexHessian::Pointer & filter, const typename TComplexImage::RegionType & extractionRegion,
		const typename TComplexImage::SizeType & padSize, const typename TComplexImage::SizeType & padLowerBound,
		typename THessian::Pointer & hessian, typename THessian::Pointer & shrinked, typename TComplexImage::Pointer & conjugated) {

	typedef ttt::MultiplyByTensorImageFilter<TComplexImage, TComplexHessian> HessianFrequencyFilterType;

	typedef ttt::HessianIFFTImageFilter<TComplexHessian, THessian> HessianIFFTFilterType;
	typedef itk::ExtractImageFilter<THessian, THessian> ExtractFilterType;
	typedef ttt::HessianFFTImageFilter<THessian, TComplexHessian> HessianFFTFilterType;

	typedef itk::AddImageFilter<THessian> AddHessianType;
	typedef itk::SubtractImageFilter<THessian> SubHessianType;

	typedef ttt::HessianShrinkImageFilter<THessian> HessianShrinkFilterType;
	typedef ttt::MultiplyTensorByConjugateTensorImageFilter<TComplexHessian> ConjugateHessianFrequencyFilterType;

	typedef ttt::ReduceTensorImageFilter<TComplexHessian, TComplexImage> HessianReducerFilterType;

	typename HessianFrequencyFilterType::Pointer m_HessianFrequencyEstimationFilter1;

	typename HessianIFFTFilterType::Pointer m_HessianIFFTFilter1;
	typename AddHessianType::Pointer m_AddHessianFilter1;

	typename HessianShrinkFilterType::Pointer m_HessianShrinkFilter1;

	typename SubHessianType::Pointer m_SubHessianFilter1;
	typename HessianFFTFilterType::Pointer m_HessianFFTFilter1;

	typename ConjugateHessianFrequencyFilterType::Pointer m_ConjugateHessianFilter1;

	typename HessianReducerFilterType::Pointer m_HessianReduceFilter1;

	typename ExtractFilterType::Pointer m_Extractor;

	m_HessianFrequencyEstimationFilter1 = HessianFrequencyFilterType::New();
	m_HessianFrequencyEstimationFilter1->SetInput1(frame);
	m_HessianFrequencyEstimationFilter1->SetInput2(filter);
	//m_HessianFrequencyEstimationFilter1->Update();
	//m_HessianFrequencyEstimationFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	//typename TComplexHessian::Pointer hessianFrequency=m_HessianFrequencyEstimationFilter1->GetOutput();
	//WriteFile<TComplexImage>(std::string("/home/morgan/tmpfiles/"),std::string("frame"),std::string(""),std::string("ome.tif"),frame);
	//WriteFile<THessian>(std::string("/home/morgan/tmpfiles/"),std::string("lagrange"),"","ome.tif",lagrange);
	//WriteFile<TComplexHessian>(std::string("/home/morgan/tmpfiles/"),std::string("filter"),"","ome.tif",filter);
	//WriteFile<TComplexHessian>(std::string("/home/morgan/tmpfiles/"),std::string("hessianFrequency"),"","ome.tif",hessianFrequency);

	typename TComplexHessian::Pointer hessianFrequencyEstimate = m_HessianFrequencyEstimationFilter1->GetOutput();
	m_HessianIFFTFilter1 = HessianIFFTFilterType::New();
	m_HessianIFFTFilter1->SetInput(hessianFrequencyEstimate);
	m_HessianIFFTFilter1->Update();

	m_Extractor = ExtractFilterType::New();
	m_Extractor->SetInput(m_HessianIFFTFilter1->GetOutput());
	m_Extractor->SetExtractionRegion(extractionRegion);
	m_Extractor->InPlaceOn();
	m_Extractor->ReleaseDataFlagOn();
	m_Extractor->Update();

//	hessian = m_Extractor->GetOutput();
//	hessian->DisconnectPipeline();

	//WriteFile<THessian>(std::string("/home/morgan/tmpfiles/"),std::string("hessian"),"","ome.tif",hessian);
	//exit(-1);

	//m_HessianIFFTFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_AddHessianFilter1 = AddHessianType::New();
	m_AddHessianFilter1->SetInput1(m_Extractor->GetOutput());
	m_AddHessianFilter1->SetInput2(lagrange);
	m_AddHessianFilter1->InPlaceOn();
	m_AddHessianFilter1->ReleaseDataFlagOn();
	//m_AddHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_HessianShrinkFilter1 = HessianShrinkFilterType::New();
	//m_HessianShrinkFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_HessianShrinkFilter1->SetInput(m_AddHessianFilter1->GetOutput());
	m_HessianShrinkFilter1->GetFunctor().m_Lambda = 1; //m_Lambda;
	m_HessianShrinkFilter1->InPlaceOn();
	m_HessianShrinkFilter1->ReleaseDataFlagOn();
	m_HessianShrinkFilter1->Update();
	shrinked = m_HessianShrinkFilter1->GetOutput();
	shrinked->DisconnectPipeline();

	m_SubHessianFilter1 = SubHessianType::New();
	m_SubHessianFilter1->SetInput1(m_HessianShrinkFilter1->GetOutput());
	m_SubHessianFilter1->SetInput2(lagrange);

	m_SubHessianFilter1->ReleaseDataFlagOn();
	m_SubHessianFilter1->Update();

	typename THessian::Pointer tmp = m_SubHessianFilter1->GetOutput();

	//m_SubHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	typename THessian::Pointer paddedTmp;
	PadImage<THessian>(tmp, padSize, padLowerBound, paddedTmp);

	m_HessianFFTFilter1 = HessianFFTFilterType::New();
	m_HessianFFTFilter1->SetInput(paddedTmp);
	m_HessianFFTFilter1->Update();
	//m_HessianFFTFilter1->SetInput(m_SubHessianFilter1->GetOutput());
	//m_HessianFFTFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_ConjugateHessianFilter1 = ConjugateHessianFrequencyFilterType::New();
	m_ConjugateHessianFilter1->SetInput1(m_HessianFFTFilter1->GetOutput());
	m_ConjugateHessianFilter1->SetInput2(filter);

	//m_ConjugateHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_HessianReduceFilter1 = HessianReducerFilterType::New();
	m_HessianReduceFilter1->SetInput(m_ConjugateHessianFilter1->GetOutput());
	//m_HessianReduceFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	//m_HessianShrinkFilter1->Update();

	m_HessianReduceFilter1->Update();

	conjugated = m_HessianReduceFilter1->GetOutput();
	conjugated->DisconnectPipeline();

}

template<class TInputImage>
typename TInputImage::SizeType GetPadSize(const typename TInputImage::Pointer & input, const typename TInputImage::Pointer & psf) {

	typename TInputImage::SizeType inputSize = input->GetLargestPossibleRegion().GetSize();
	typename TInputImage::SizeType kernelSize = psf->GetLargestPossibleRegion().GetSize();

	typename TInputImage::SizeType padSize { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		padSize[i] = inputSize[i] + kernelSize[i];
		// Use the valid sizes for VNL because they are fast sizes for
		// both VNL and FFTW.
		while (!VnlFFTCommon::IsDimensionSizeLegal(padSize[i])) {
			padSize[i]++;
		}
	}

	return padSize;
}

template<class TInputImage>
typename TInputImage::SizeType GetPadLowerBound(const typename TInputImage::Pointer & input, const typename TInputImage::Pointer & kernel) {
	typename TInputImage::SizeType inputSize = input->GetLargestPossibleRegion().GetSize();
	typename TInputImage::SizeType padSize = GetPadSize<TInputImage>(input, kernel);

	typename TInputImage::SizeType inputLowerBound { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		inputLowerBound[i] = (padSize[i] - inputSize[i]) / 2;
	}

	return inputLowerBound;
}

template<class TInputImage, class TComplexImage> void PrepareImageAndPSF(const typename TInputImage::Pointer & frame,
		const typename TInputImage::Pointer & psf, typename TInputImage::Pointer & padded, typename TComplexImage::Pointer & transformed,
		typename TInputImage::Pointer & paddedPSF, typename TComplexImage::Pointer & transferFunction) {
	typename TInputImage::SizeType padSize = GetPadSize<TInputImage>(frame, psf);
	typename TInputImage::SizeType padLowerBound = GetPadLowerBound<TInputImage>(frame, psf);

	PadImage<TInputImage>(frame, padSize, padLowerBound, padded);

	ImageFFT<TInputImage, TComplexImage>(padded, transformed);
	////////////////

	typename TInputImage::RegionType kernelRegion = psf->GetLargestPossibleRegion();
	typename TInputImage::SizeType kernelSize = kernelRegion.GetSize();

	typename TInputImage::SizeType kernelUpperBound { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		kernelUpperBound[i] = padSize[i] - kernelSize[i];
	}

	typename TInputImage::Pointer paddedKernelImage = ITK_NULLPTR;

	typedef itk::NormalizeToConstantImageFilter<TInputImage, TInputImage> NormalizeFilterType;
	typename NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
	normalizeFilter->SetConstant(itk::NumericTraits<typename TInputImage::PixelType>::OneValue());
	//normalizeFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	normalizeFilter->SetInput(psf);
	normalizeFilter->ReleaseDataFlagOn();
	//progress->RegisterInternalFilter( normalizeFilter,
	//                                  0.2f * paddingWeight * progressWeight );

	// Pad the kernel image with zeros.
	typedef itk::ConstantPadImageFilter<TInputImage, TInputImage> KernelPadType;
	typedef typename KernelPadType::Pointer KernelPadPointer;
	KernelPadPointer kernelPadder = KernelPadType::New();
	kernelPadder->SetConstant(itk::NumericTraits<typename TInputImage::PixelType>::ZeroValue());
	kernelPadder->SetPadUpperBound(kernelUpperBound);
	//kernelPadder->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelPadder->SetInput(normalizeFilter->GetOutput());
	kernelPadder->ReleaseDataFlagOn();
	//progress->RegisterInternalFilter( kernelPadder, 0.8f * paddingWeight * progressWeight );
	paddedKernelImage = kernelPadder->GetOutput();

	// Shift the padded kernel image.
	typedef itk::CyclicShiftImageFilter<TInputImage, TInputImage> KernelShiftFilterType;
	typename KernelShiftFilterType::Pointer kernelShifter = KernelShiftFilterType::New();
	typename KernelShiftFilterType::OffsetType kernelShift { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		kernelShift[i] = -(kernelSize[i] / 2);
	}
	kernelShifter->SetShift(kernelShift);
	//kernelShifter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelShifter->SetInput(paddedKernelImage);
	//kernelShifter->ReleaseDataFlagOn();
	//progress->RegisterInternalFilter( kernelShifter, 0.1f * progressWeight );
	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TInputImage, TComplexImage> FFTFilterType;
	typename FFTFilterType::Pointer kernelFFTFilter = FFTFilterType::New();
	//kernelFFTFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelFFTFilter->SetInput(kernelShifter->GetOutput());
	//progress->RegisterInternalFilter( kernelFFTFilter, 0.699f * progressWeight );
	kernelFFTFilter->Update();

	paddedPSF = kernelShifter->GetOutput();
	paddedPSF->DisconnectPipeline();

	typedef itk::ChangeInformationImageFilter<TComplexImage> InfoFilterType;
	typename InfoFilterType::Pointer kernelInfoFilter = InfoFilterType::New();
	kernelInfoFilter->ChangeRegionOn();

	typedef typename InfoFilterType::OutputImageOffsetValueType InfoOffsetValueType;
	//InputSizeType inputLowerBound = this->GetPadLowerBound();
	typename TInputImage::IndexType inputIndex = frame->GetLargestPossibleRegion().GetIndex();
	typename TInputImage::IndexType kernelIndex = psf->GetLargestPossibleRegion().GetIndex();
	InfoOffsetValueType kernelOffset[TInputImage::ImageDimension] { };
	for (int i = 0; i < TInputImage::ImageDimension; ++i) {
		kernelOffset[i] = static_cast<InfoOffsetValueType>(inputIndex[i] - padLowerBound[i] - kernelIndex[i]);
	}
	kernelInfoFilter->SetOutputOffset(kernelOffset);
	//kernelInfoFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelInfoFilter->SetInput(kernelFFTFilter->GetOutput());
	//progress->RegisterInternalFilter( kernelInfoFilter, 0.001f * progressWeight );
	kernelInfoFilter->Update();

	transferFunction = kernelInfoFilter->GetOutput();
	transferFunction->DisconnectPipeline();

}

template<class TImage> void AdjustImage(const typename TImage::Pointer & image, const typename TImage::OffsetType & offset,
		typename TImage::Pointer & adjusted) {
	typedef itk::ChangeInformationImageFilter<TImage> InfoFilterType;
	typename InfoFilterType::Pointer infoFilter = InfoFilterType::New();
	infoFilter->ChangeRegionOn();
#if 0
	typedef typename itk::ChangeInformationImageFilter< TImage >::OutputImageOffsetValueType InfoOffsetValueType;
	InfoOffsetValueType offset[TImage::ImageDimension] {};

	for (int i = 0; i < TImage::ImageDimension; ++i) {
		offset[i] = static_cast< InfoOffsetValueType >( size[i]-image->GetLargestPossibleRegion().GetSize(i))/2;
	}
#endif
	infoFilter->SetOutputOffset(offset);
	infoFilter->SetInput(image);
	infoFilter->Update();
	adjusted = infoFilter->GetOutput();
	adjusted->DisconnectPipeline();

}


template<class TImage, class TComplexImage> void DoPoissonStage(const typename TImage::Pointer & original,
		const typename TComplexImage::Pointer & estimate, const typename TImage::Pointer & lagrange, const typename TComplexImage::Pointer & psf,
		const typename TImage::SizeType & padSize, const typename TImage::SizeType & padLowerBound, typename TImage::Pointer & denoised,
		typename TComplexImage::Pointer & conjugated) {

	//double alpha { 16.0 };
	double alpha { ALPHAHESSIAN };

	typedef itk::MultiplyImageFilter<TComplexImage> ComplexMultiplyType;
	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TImage, TComplexImage> FFTFilterType;
	typedef itk::HalfHermitianToRealInverseFFTImageFilter<TComplexImage, TImage> IFFTFilterType;

	typedef itk::ExtractImageFilter<TImage, TImage> ExtractFilterType;

	typedef itk::AddImageFilter<TImage> AddType;
	typedef ttt::PoissonShrinkImageFilter<TImage> PoissonShrinkFilterType;
	typedef itk::SubtractImageFilter<TImage> SubType;
	typedef itk::MultiplyImageFilter<TComplexImage, itk::ComplexConjugateImageAdaptor<TComplexImage> > ComplexConjugateMultiplyType;
	typedef itk::ComplexConjugateImageAdaptor<TComplexImage> ConjugateAdaptor;

	typename ComplexMultiplyType::Pointer m_ComplexMultiplyFilter1 { };
	typename IFFTFilterType::Pointer m_IFFTFilter1 { };

	typename ExtractFilterType::Pointer m_Extractor { };

	typename AddType::Pointer m_AddFilter1 { };
	typename PoissonShrinkFilterType::Pointer m_PoissonShrinkFilter1 { };

	typename SubType::Pointer m_SubFilter1 { };
	typename FFTFilterType::Pointer m_FFTFilter2 { };
	typename ComplexConjugateMultiplyType::Pointer m_ComplexConjugateMultiplyFilter2 { };

	typename ConjugateAdaptor::Pointer m_ConjugateTransferFunction { };

	m_ComplexMultiplyFilter1 = ComplexMultiplyType::New();
	m_ComplexMultiplyFilter1->ReleaseDataFlagOn();
	m_ComplexMultiplyFilter1->SetInput1(estimate);
	m_ComplexMultiplyFilter1->SetInput2(psf);
	m_ComplexMultiplyFilter1->Update();

	m_IFFTFilter1 = IFFTFilterType::New();

	m_IFFTFilter1->SetInput(m_ComplexMultiplyFilter1->GetOutput());

	m_IFFTFilter1->ReleaseDataFlagOn();
	//m_IFFTFilter1->Update();

	m_Extractor = ExtractFilterType::New();
	m_Extractor->SetInput(m_IFFTFilter1->GetOutput());
	m_Extractor->SetExtractionRegion(original->GetLargestPossibleRegion());

	m_AddFilter1 = AddType::New();
	m_AddFilter1->SetInput1(m_Extractor->GetOutput());
	m_AddFilter1->SetInput2(lagrange);
	m_AddFilter1->ReleaseDataFlagOn();
	m_AddFilter1->Update();

	typename TImage::Pointer added = m_AddFilter1->GetOutput();

	//typename TImage::Pointer added = m_AddFilter1->GetOutput();
	m_PoissonShrinkFilter1 = PoissonShrinkFilterType::New();
	m_PoissonShrinkFilter1->SetInput1(added);
	m_PoissonShrinkFilter1->SetInput2(original);
	m_PoissonShrinkFilter1->GetFunctor().m_Alpha = alpha;
	m_PoissonShrinkFilter1->ReleaseDataFlagOn();
	m_PoissonShrinkFilter1->Update();
	denoised = m_PoissonShrinkFilter1->GetOutput();
	denoised->DisconnectPipeline();

	m_SubFilter1 = SubType::New();
	m_SubFilter1->SetInput1(m_PoissonShrinkFilter1->GetOutput());
	m_SubFilter1->SetInput2(lagrange);

	m_SubFilter1->ReleaseDataFlagOn();
	m_SubFilter1->Update();

	typename TImage::Pointer tmp = m_SubFilter1->GetOutput();
	PadImage<TImage>(tmp, padSize, padLowerBound, tmp);

	m_FFTFilter2 = FFTFilterType::New();
	m_FFTFilter2->SetInput(tmp);
	m_FFTFilter2->ReleaseDataFlagOn();

	m_ConjugateTransferFunction = ConjugateAdaptor::New();
	m_ConjugateTransferFunction->SetImage(psf);

	m_ComplexConjugateMultiplyFilter2 = ComplexConjugateMultiplyType::New();
	m_ComplexConjugateMultiplyFilter2->SetInput1(m_FFTFilter2->GetOutput());
	m_ComplexConjugateMultiplyFilter2->SetInput2(m_ConjugateTransferFunction);
	m_ComplexConjugateMultiplyFilter2->ReleaseDataFlagOn();
	m_ComplexConjugateMultiplyFilter2->Update();

	conjugated = m_ComplexConjugateMultiplyFilter2->GetOutput();
	conjugated->DisconnectPipeline();

#if 0
	typedef ttt::DeconvolutionPoissonStageImageFilter<TComplexImage,TImage> PoissonStageType;

	typename PoissonStageType::Pointer poissonStage = PoissonStageType::New();

	poissonStage->SetCurrentEstimation(estimate);
	poissonStage->SetInputImage(original);
	poissonStage->SetTransferFunction(psf);
	poissonStage->SetLagrangeMultiplier(lagrange);
	poissonStage->Update();
	denoised=poissonStage->GetShrinkedImage();
	conjugated=poissonStage->GetConjugatedImage();
#endif

}



template<class TProject> class HSPVRRAL {
	typedef typename TProject::ImageType InputImageType;
	typedef typename TProject::ComplexImageType ComplexImageType;
	typedef typename TProject::FieldImageType FieldType;

	typedef typename TProject::FloatType FloatType;
	//typedef itk::Image<std::complex<FloatType>, 3> ComplexImageType;
	//typedef itk::Image<itk::Vector<FloatType, 3>, 3> FieldType;

	typedef itk::SymmetricSecondRankTensor<FloatType, 3> HessianType;
	typedef itk::Image<HessianType, 3> HessianImageType;
	typedef itk::Image<itk::SymmetricSecondRankTensor<std::complex<FloatType>, 3>, 3> HessianComplexImageType;

public:
	HSPVRRAL(TProject & project) :
		m_Project(project)
	{
		m_NumberOfLevels=1;

		m_NumberOfIterations=10;
		m_CurrentLevel=0;
		m_CurrentIteration=0;
		m_Restart=false;

		m_NumberOfDeconvolutionIterations=3;
		m_NumberOfBlindIterations=5;
		//Pyramid

		//m_Factors[0]= 8;
		//m_Factors[1]= 8;
		//m_Factors[2]= 1;

		m_Factors[0]= 1;
		m_Factors[1]= 1;
		m_Factors[2]= 1;

		m_W=1;
		m_FirstRegistration=true;
	}

	virtual ~HSPVRRAL(){

	}
	virtual void InitPyramid(){
		typename InputImageType::Pointer psf;

		//psf=m_Project.GetTemplatePSF();

		typename InputImageType::SizeType psfSize;
		psfSize.Fill(15);
		psf = GenerateIdentityPSF<InputImageType>(psfSize);


		typedef itk::MultiResolutionPyramidImageFilter<InputImageType, InputImageType> RecursiveMultiResolutionPyramidImageFilterType;
		typename RecursiveMultiResolutionPyramidImageFilterType::Pointer psfMultiResolutionPyramidImageFilter =
					RecursiveMultiResolutionPyramidImageFilterType::New();
			psfMultiResolutionPyramidImageFilter->SetInput(psf);
			psfMultiResolutionPyramidImageFilter->SetNumberOfLevels(m_NumberOfLevels);
			psfMultiResolutionPyramidImageFilter->SetStartingShrinkFactors(m_Factors);
			psfMultiResolutionPyramidImageFilter->Update();

			for(unsigned int t=0;t<m_Project.GetNumberOfFrames();t++){
				typename InputImageType::Pointer frame;
				frame = m_Project.GetOriginalImage(t);
				typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> ScalerType;
				typename ScalerType::Pointer scaler = ScalerType::New();


				scaler->SetInput1(frame);
				//scaler->SetConstant2(1.0);
				scaler->SetConstant2(65536.0);
				scaler->Update();

				frame = scaler->GetOutput();
				frame->DisconnectPipeline();

				typename RecursiveMultiResolutionPyramidImageFilterType::Pointer recursiveMultiResolutionPyramidImageFilter =
						RecursiveMultiResolutionPyramidImageFilterType::New();
				recursiveMultiResolutionPyramidImageFilter->SetInput(frame);
				recursiveMultiResolutionPyramidImageFilter->SetNumberOfLevels(m_NumberOfLevels);
				recursiveMultiResolutionPyramidImageFilter->SetStartingShrinkFactors(m_Factors);
				recursiveMultiResolutionPyramidImageFilter->Update();


				for (int i = 0; i < m_NumberOfLevels; i++) {

					std::cout << recursiveMultiResolutionPyramidImageFilter->GetOutput(i)->GetLargestPossibleRegion().GetSize() << std::endl;
					std::cout << recursiveMultiResolutionPyramidImageFilter->GetOutput(i)->GetSpacing() << std::endl;

					m_Project.SetObservedImage(t, i, recursiveMultiResolutionPyramidImageFilter->GetOutput(i));
					m_Project.SetPSF(t, i, psfMultiResolutionPyramidImageFilter->GetOutput(i));
					//m_Project.SetPSF(t, i, psf);
					m_Project.SetEstimatedImage(t, i, recursiveMultiResolutionPyramidImageFilter->GetOutput(i));
				}
			}

	}
	virtual void DoRegistration(){
		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {

			int moving { t };
			typename InputImageType::Pointer movingFrame { };
			movingFrame = m_Project.GetEstimatedImage(moving, m_CurrentLevel);
			auto origin = movingFrame->GetOrigin();
			origin.Fill(0);
			movingFrame->SetOrigin(origin);

			for (unsigned int k = 1; k <= m_W; k++) {

				int fixed =int( t + k );

				if (fixed < 0 || fixed >= m_Project.GetNumberOfFrames())
					continue;

				typename InputImageType::Pointer fixedFrame { };
				fixedFrame = m_Project.GetEstimatedImage(fixed, m_CurrentLevel);
				//firstFrame = project.GetOriginalImage(fixed, level);
				auto origin = movingFrame->GetOrigin();
				origin.Fill(0);
				fixedFrame->SetOrigin(origin);
				//ReadFile<InputImageType>(std::string(directory),std::string("estimate"),firstFrameNum.str(),"mha",firstFrame);
				//typename InputImageType::Pointer firstFrame=ReadFile<InputImageType>(std::string(directory),std::string(prefix),firstFrameNum.str(),"ome.tif");

				std::cout << fixed << " to " << moving << std::endl;

#if 0
				typedef itk::LevelSetMotionRegistrationFilter<InputImageType,InputImageType,FieldType> RegistrationFilterType;
				typename RegistrationFilterType::Pointer registrationFilter = RegistrationFilterType::New();

				registrationFilter->SetFixedImage(fixedFrame);
				registrationFilter->SetMovingImage(movingFrame);


				if (!m_FirstRegistration) {

					typename FieldType::Pointer initialField { };
					initialField = m_Project.GetMotionField(fixed, moving, m_CurrentLevel);

					registrationFilter->SetInitialDisplacementField(initialField);
				}
				registrationFilter->Update();
				typename FieldType::Pointer directDisplacementField = registrationFilter->GetDisplacementField();
				directDisplacementField->DisconnectPipeline();
				m_Project.SetMotionField(fixed, moving, m_CurrentLevel,directDisplacementField);

				//registrationFilter->SmoothDisplacementFieldOff();
				//registrationFilter->SetStandardDeviations(0.1);
				registrationFilter->SetNumberOfIterations( m_FirstRegistration ? 20 : 5 );

				registrationFilter->SetFixedImage(movingFrame);
				registrationFilter->SetMovingImage(fixedFrame);


				if (!m_FirstRegistration) {

					typename FieldType::Pointer initialField { };
					initialField = m_Project.GetMotionField(moving, fixed, m_CurrentLevel);

					registrationFilter->SetInitialDisplacementField(initialField);
				}
				registrationFilter->Update();
				typename FieldType::Pointer inverseDisplacementField = registrationFilter->GetDisplacementField();
				inverseDisplacementField->DisconnectPipeline();
				m_Project.SetMotionField(moving,fixed,m_CurrentLevel,inverseDisplacementField);
#endif

#if 1
				typename FieldType::Pointer directDisplacementField { }, inverseDisplacementField { }, velocityField { };

				///////////////////////////////

				int numberOfIterations { m_FirstRegistration ? 20 : 10 };

				//int numberOfIterations { 1 };
				//int numberOfIterations {3};
				int numberOfLevels { 1 };
				int numberOfExponentiatorIterations { 4 };
				double timestep { 1.0 };

				bool useImageSpacing { true };

				// Regularizer parameters
				int regularizerType { 1 };       // Diffusive
				float regulAlpha = 0.5;
				float regulVar = 0.5;
				float regulMu = 0.5;
				float regulLambda = 0.5;

				int nccRadius { 2 };

				// Force parameters
				int forceType { 0 };              // Demon
				int forceDomain { 0 };            // Warped moving

				// Stop criterion parameters
				int stopCriterionPolicy { 1 }; // Simple graduated is default
				float stopCriterionSlope = 0.005;

				std::vector<unsigned int> its(numberOfLevels);
				its[numberOfLevels - 1] = numberOfIterations;
				for (int level = numberOfLevels - 2; level >= 0; --level) {
					its[level] = its[level + 1];
				}

				//if(level==0 && it==0){
				//typedef itk::VariationalRegistrationNCCFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				//typedef ttt::VariationalRegistrationLogNCCFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				//typedef itk::VariationalRegistrationDemonsFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				typedef itk::VariationalRegistrationSSDFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				typename NCCFunctionType::Pointer function = NCCFunctionType::New();
				function->SetGradientTypeToWarpedMovingImage();
				function->SetTimeStep(100.0);

				//regFilter->SetNumberOfExponentiatorIterations(numberOfExponentiatorIterations);

				typedef itk::VariationalRegistrationDiffusionRegularizer<FieldType> DiffusionRegularizerType;
				//typedef itk::VariationalRegistrationElasticRegularizer<FieldType> DiffusionRegularizerType;
				//typedef itk::VariationalRegistrationGaussianRegularizer<FieldType> DiffusionRegularizerType;

				typename DiffusionRegularizerType::Pointer regularizer = DiffusionRegularizerType::New();
				regularizer->SetAlpha(0.5);
				regularizer->SetUseImageSpacing(true);
				regularizer->InPlaceOff();


				//typedef itk::VariationalSymmetricDiffeomorphicRegistrationFilter<InputImageType, InputImageType, FieldType> VariationalFilterType;
				typedef itk::VariationalRegistrationFilter<InputImageType, InputImageType, FieldType> VariationalFilterType;
				//typedef itk::VariationalDiffeomorphicRegistrationFilter<InputImageType, InputImageType, FieldType> VariationalFilterType;

				typename VariationalFilterType::Pointer regFilter = VariationalFilterType::New();


				//regularizer->SetAlpha(0);
				//regularizer->SetAlpha(regularizer->GetAlpha() * pow(2,6));
				//regularizer->SetAlpha()
				//regularizer->SetStandardDeviations(25);
				//regularizer->SetMaximumKernelWidth(75);
				//regularizer->SetUseImageSpacing(false);
				regFilter->SetRegularizer(regularizer);
				//regFilter->SmoothDisplacementFieldOff();
				//regFilter->SmoothUpdateFieldOff();


				//function->SetTimeStep(100*function->GetTimeStep());
#if 0
				typename NCCFunctionType::RadiusType radius;
				radius.Fill(4);
				function->SetRadius(radius);
#endif
				regFilter->SetDifferenceFunction(function);

				typedef itk::VariationalRegistrationMultiResolutionFilter<InputImageType, InputImageType, FieldType> MRRegistrationFilterType;

				typename MRRegistrationFilterType::Pointer mrRegFilter = MRRegistrationFilterType::New();
				mrRegFilter->SetRegistrationFilter(regFilter);
				mrRegFilter->SetMovingImage(movingFrame);
				mrRegFilter->SetFixedImage(fixedFrame);

				mrRegFilter->SetNumberOfLevels(numberOfLevels);
				mrRegFilter->SetNumberOfIterations(its.data());

				if (!m_FirstRegistration) {

					typename FieldType::Pointer initialField { };
					initialField = m_Project.GetMotionField(moving, fixed, m_CurrentLevel);

					mrRegFilter->SetInitialField(initialField);
				}

				//
				// Setup stop criterion
				//
				typedef VariationalRegistrationStopCriterion<VariationalFilterType, MRRegistrationFilterType> StopCriterionType;
				typename StopCriterionType::Pointer stopCriterion = StopCriterionType::New();
				stopCriterion->SetRegressionLineSlopeThreshold(stopCriterionSlope);
				stopCriterion->PerformLineFittingMaxDistanceCheckOn();

				switch (stopCriterionPolicy) {
				case 1:
					stopCriterion->SetMultiResolutionPolicyToSimpleGraduated();
					break;
				case 2:
					stopCriterion->SetMultiResolutionPolicyToGraduated();
					break;
				default:
					stopCriterion->SetMultiResolutionPolicyToDefault();
					break;
				}

				regFilter->AddObserver(itk::IterationEvent(), stopCriterion);
				mrRegFilter->AddObserver(itk::IterationEvent(), stopCriterion);
				mrRegFilter->AddObserver(itk::InitializeEvent(), stopCriterion);
				//
				// Setup logger
				//
				typedef VariationalRegistrationLogger<VariationalFilterType, MRRegistrationFilterType> LoggerType;
				typename LoggerType::Pointer logger = LoggerType::New();

				regFilter->AddObserver(itk::IterationEvent(), logger);
				mrRegFilter->AddObserver(itk::IterationEvent(), logger);

				mrRegFilter->Update();

				directDisplacementField = mrRegFilter->GetDisplacementField();
				directDisplacementField->DisconnectPipeline();
				m_Project.SetMotionField(moving,fixed, m_CurrentLevel, directDisplacementField);

				mrRegFilter->SetMovingImage(fixedFrame);
				mrRegFilter->SetFixedImage(movingFrame);

				if (!m_FirstRegistration) {

					typename FieldType::Pointer initialField { };
					initialField = m_Project.GetMotionField( fixed, moving, m_CurrentLevel);

					mrRegFilter->SetInitialField(initialField);
				}

				mrRegFilter->Update();
#if 0
				typedef itk::SubtractImageFilter<FieldType,FieldType,FieldType> SubtractorType;
				typename SubtractorType::Pointer substractor = SubtractorType::New();
				substractor->SetConstant1(0.0);
				substractor->SetInput2(directDisplacementField);
				substractor->Update();
				inverseDisplacementField = substractor->GetOutput();
#endif
#if 0
				inverseDisplacementField= regFilter->GetInverseDisplacementField();
				inverseDisplacementField->DisconnectPipeline();
#endif
				inverseDisplacementField= mrRegFilter->GetDisplacementField();
				inverseDisplacementField->DisconnectPipeline();
				m_Project.SetMotionField(fixed,moving, m_CurrentLevel, inverseDisplacementField);


#endif
			} //Do registration
		}
		m_FirstRegistration = false;
	}
	void InitLevelIteration(){
		//firstRegistration = true;

		typename InputImageType::Pointer original;
		original = m_Project.GetObservedImage(0, m_CurrentLevel);


		this->m_OneSpacing.Fill(0);
		this->m_OriginalSpacing=original->GetSpacing();

		this->m_OneSpacing.Fill(1);

		this->m_OriginalOrigin = original->GetOrigin();
		this->m_ZeroOrigin.Fill(0);

		typename InputImageType::Pointer psf = m_Project.GetPSF(0, m_CurrentLevel);

		this->m_PadSize= GetPadSize<InputImageType>(original, psf);
		this->m_PadLowerBound= GetPadLowerBound<InputImageType>(original, psf);



		this->m_Offset[0] = -this->m_PadLowerBound[0];
		this->m_Offset[1] = -this->m_PadLowerBound[1];
		this->m_Offset[2] = -this->m_PadLowerBound[2];

		typedef ttt::CentralDifferenceHessianSource<FloatType, 3> HessianSourceType;
		//typedef ttt::GaussianHessianSource<FloatType, 3> HessianSourceType;

		typename HessianSourceType::Pointer hessianSource = HessianSourceType::New();
		//hessianSource->SetNumberOfThreads(this->GetNumberOfThreads());
		hessianSource->SetLowerPad(m_PadLowerBound);
		hessianSource->SetPadSize(m_PadSize);

		typename InputImageType::SpacingType hessianSpacing;

		hessianSpacing[0] = 1;
		hessianSpacing[1] = 1;
		hessianSpacing[2] = original->GetSpacing()[2] / original->GetSpacing()[0];

		hessianSource->SetSpacing(hessianSpacing);
		hessianSource->Update();

		this->m_HessianFilter = hessianSource->GetOutput();
		this->m_HessianFilter->DisconnectPipeline();

		typedef ttt::TensorToEnergyImageFilter<HessianComplexImageType, InputImageType> HessianEnergyFilter;
		typename HessianEnergyFilter::Pointer hessianEnergyFilter1 = HessianEnergyFilter::New();
		hessianEnergyFilter1->SetInput(this->m_HessianFilter);
		hessianEnergyFilter1->Update();
		this->m_HessianNormalizer = hessianEnergyFilter1->GetOutput();
		this->m_HessianNormalizer->DisconnectPipeline();

		for (int t = 0; t <m_Project.GetNumberOfFrames(); t++) {


			typename InputImageType::Pointer frame;

			frame = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename InputImageType::Pointer paddedFrame { }, paddedPSF { }, psf { };
			typename ComplexImageType::Pointer transformedFrame { };
			typename ComplexImageType::Pointer transferFunction { };
			psf = m_Project.GetPSF(t, m_CurrentLevel);

			PrepareImageAndPSF<InputImageType, ComplexImageType>(frame, psf, paddedFrame, transformedFrame, paddedPSF, transferFunction);

			m_Project.SetEstimatedFrequencyImage(t, m_CurrentLevel, transformedFrame);
			m_Project.SetTransferImage(t, m_CurrentLevel, transferFunction);
			//WriteFile<InputImageType>(std::string(directory),std::string("originalPSF"),frameNum.str(),"mha",paddedPSF);



		}

	}
	void InitDeconvolutionIteration() {


		for (int t = 0; t <m_Project.GetNumberOfFrames(); t++) {

			typename InputImageType::Pointer frame;

			frame = m_Project.GetEstimatedImage(t, m_CurrentLevel);



			typename InputImageType::Pointer poissonLagrange = InputImageType::New();
			poissonLagrange->CopyInformation(frame);
			poissonLagrange->SetRegions(frame->GetLargestPossibleRegion());

			poissonLagrange->Allocate();
			poissonLagrange->FillBuffer(0.0);

			m_Project.SetPoissonLagrange(t, m_CurrentLevel, poissonLagrange);
			m_Project.SetBoundsLagrange(t, m_CurrentLevel, poissonLagrange);

			typename HessianImageType::Pointer hessianLagrange = HessianImageType::New();

			hessianLagrange->CopyInformation(frame);
			hessianLagrange->SetRegions(frame->GetLargestPossibleRegion());

			hessianLagrange->Allocate();
			hessianLagrange->FillBuffer(itk::NumericTraits<HessianType>::ZeroValue());

			m_Project.SetHessianLagrange(t, m_CurrentLevel, hessianLagrange);

			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {
				if(t1<0 || t1==t || t1>=m_Project.GetNumberOfFrames()) continue;
					m_Project.SetMovingLagrange(t1, t, m_CurrentLevel, poissonLagrange);
					m_Project.SetWarpedPoissonLagrange(t1, t, m_CurrentLevel, poissonLagrange);

			}
		} //DO INIT
	}

	void DoShrinkStep() {
		auto original = m_Project.GetObservedImage(0, m_CurrentLevel);

		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;
			std::cout << "SHRINK T " << t << std::endl;

			typename InputImageType::Pointer original;
			original = m_Project.GetObservedImage(t, m_CurrentLevel);

			typename InputImageType::Pointer estimate;
			estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename ComplexImageType::Pointer transfer;
			transfer = m_Project.GetTransferImage(t, m_CurrentLevel);

			AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);
			typename ComplexImageType::Pointer estimateFrequency { };

			typename InputImageType::Pointer paddedEstimate { };

			//typename InputImageType::SizeType padSize = GetPadSize<InputImageType>(original, psf);
			//typename InputImageType::SizeType padLowerBound = GetPadLowerBound<InputImageType>(original, psf);

			//estimateFrequency = project.GetEstimatedFrequencyImage(t,level);
			PadImage<InputImageType>(estimate, m_PadSize, m_PadLowerBound, paddedEstimate);

			ImageFFT<InputImageType, ComplexImageType>(paddedEstimate, estimateFrequency);

			typename InputImageType::Pointer poissonLagrange;

			poissonLagrange = m_Project.GetPoissonLagrange(t, m_CurrentLevel);
			//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);

			typename InputImageType::Pointer denoised { };
			typename InputImageType::Pointer conjugatedDenoised { };
			typename ComplexImageType::Pointer conjugatedDenoisedFrequency { };

			original->SetSpacing(m_OneSpacing);
			poissonLagrange->SetSpacing(m_OneSpacing);
			transfer->SetSpacing(m_OneSpacing);
			estimateFrequency->SetSpacing(m_OneSpacing);

			original->SetOrigin(m_ZeroOrigin);
			poissonLagrange->SetOrigin(m_ZeroOrigin);
			transfer->SetOrigin(m_ZeroOrigin);
			estimateFrequency->SetOrigin(m_ZeroOrigin);

			DoPoissonStage<InputImageType, ComplexImageType>(original, estimateFrequency, poissonLagrange, transfer, m_PadSize, m_PadLowerBound, denoised,
					conjugatedDenoisedFrequency);
			denoised->SetSpacing(m_OriginalSpacing);
			denoised->SetOrigin(m_OriginalOrigin);
			m_Project.SetPoissonShrinkedImage(t, m_CurrentLevel, denoised);
			//WriteFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);

			//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedDenoisedFrequency"),frameNum.str(),"mha",conjugatedDenoisedFrequency);

			ImageIFFT<InputImageType, ComplexImageType>(conjugatedDenoisedFrequency, conjugatedDenoised);
			CropImage<InputImageType>(conjugatedDenoised, original->GetLargestPossibleRegion(), conjugatedDenoised);
			conjugatedDenoised->SetSpacing(m_OriginalSpacing);
			conjugatedDenoised->SetOrigin(m_OriginalOrigin);
			m_Project.SetConjugatedPoisson(t, m_CurrentLevel, conjugatedDenoised);
			//WriteFile<InputImageType>(std::string(directory),std::string("conjugatedDenoised"),frameNum.str(),"mha",conjugatedDenoised);

#if 0
			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {
				if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
					continue;
				typename InputImageType::Pointer estimate;
				estimate = m_Project.GetEstimatedImage(t1, m_CurrentLevel);

				typedef itk::ContinuousBorderWarpImageFilter<InputImageType, InputImageType, FieldType> MovingImageWarperType;

				typename InputImageType::Pointer movingEstimate, shrinkedMoving, conjugatedMoving, lagrangeMoving;

				typename MovingImageWarperType::Pointer warper = MovingImageWarperType::New();

				typename FieldType::Pointer registrationField;
				registrationField = m_Project.GetMotionField(t1, t, m_CurrentLevel);

				warper->SetInput(estimate);
				warper->SetOutputParametersFromImage(estimate);
				warper->SetDisplacementField(registrationField);
				warper->Update();

				shrinkedMoving = warper->GetOutput();
				shrinkedMoving->DisconnectPipeline();


				PadImage<InputImageType>(estimate, m_PadSize, m_PadLowerBound, paddedEstimate);

				ImageFFT<InputImageType, ComplexImageType>(paddedEstimate, estimateFrequency);



				typename InputImageType::Pointer warpedPoissonLagrange;

				warpedPoissonLagrange = m_Project.GetWarpedPoissonLagrange(t,t1, m_CurrentLevel);
				//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);

				typename InputImageType::Pointer denoised { };
				typename InputImageType::Pointer conjugatedDenoised { };
				typename ComplexImageType::Pointer conjugatedDenoisedFrequency { };

				original->SetSpacing(m_OneSpacing);
				warpedPoissonLagrange->SetSpacing(m_OneSpacing);
				transfer->SetSpacing(m_OneSpacing);
				estimateFrequency->SetSpacing(m_OneSpacing);

				original->SetOrigin(m_ZeroOrigin);
				poissonLagrange->SetOrigin(m_ZeroOrigin);
				transfer->SetOrigin(m_ZeroOrigin);
				estimateFrequency->SetOrigin(m_ZeroOrigin);

				DoPoissonStage<InputImageType, ComplexImageType>(original, estimateFrequency, poissonLagrange, transfer, m_PadSize, m_PadLowerBound, denoised,
									conjugatedDenoisedFrequency);
				denoised->SetSpacing(m_OriginalSpacing);
				denoised->SetOrigin(m_OriginalOrigin);

				m_Project.SetWarpedPoissonShrinkedImage(t,t1, m_CurrentLevel, denoised);
				//WriteFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);

				//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedDenoisedFrequency"),frameNum.str(),"mha",conjugatedDenoisedFrequency);

				ImageIFFT<InputImageType, ComplexImageType>(conjugatedDenoisedFrequency, conjugatedDenoised);
				CropImage<InputImageType>(conjugatedDenoised, original->GetLargestPossibleRegion(), conjugatedDenoised);



				conjugatedDenoised->SetSpacing(m_OriginalSpacing);
				conjugatedDenoised->SetOrigin(m_OriginalOrigin);

				registrationField = m_Project.GetMotionField(t, t1, m_CurrentLevel);

				warper->SetInput(conjugatedDenoised);
				warper->SetOutputParametersFromImage(estimate);
				warper->SetDisplacementField(registrationField);
				warper->Update();

				conjugatedDenoised = warper->GetOutput();
				conjugatedDenoised->DisconnectPipeline();

				m_Project.SetConjugatedWarpedPoisson(t,t1, m_CurrentLevel, conjugatedDenoised);



			}
#endif
		}

		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {

			typename HessianImageType::Pointer hessianLagrange;

			hessianLagrange = m_Project.GetHessianLagrange(t, m_CurrentLevel);
			typename InputImageType::PointType originalOrigin = hessianLagrange->GetOrigin();

			typename InputImageType::SpacingType originalSpacing = hessianLagrange->GetSpacing();

			//ReadFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);

			typename InputImageType::PointType zeroOrigin;
			zeroOrigin.Fill(0.0);
			typename InputImageType::SpacingType oneSpacing;
			oneSpacing.Fill(1.0);

			hessianLagrange->SetOrigin(zeroOrigin);
			hessianLagrange->SetSpacing(oneSpacing);

			typename HessianImageType::Pointer shrinkedHessian { }, hessian { };
			typename ComplexImageType::Pointer conjugatedHessian { };
			typename ComplexImageType::Pointer estimateFrequency { };

			typename InputImageType::Pointer estimate;
			estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename InputImageType::Pointer paddedEstimate { };

			PadImage<InputImageType>(estimate, m_PadSize, m_PadLowerBound, paddedEstimate);

			ImageFFT<InputImageType, ComplexImageType>(paddedEstimate, estimateFrequency);


			DoHessianStage<ComplexImageType, HessianImageType, HessianComplexImageType>(estimateFrequency, hessianLagrange, m_HessianFilter,
					original->GetLargestPossibleRegion(), m_PadSize, m_PadLowerBound, hessian, shrinkedHessian, conjugatedHessian);

			typename InputImageType::Pointer conjugatedHessianAmplitude;

			ImageIFFT<InputImageType, ComplexImageType>(conjugatedHessian, conjugatedHessianAmplitude);
			CropImage<InputImageType>(conjugatedHessianAmplitude, original->GetLargestPossibleRegion(), conjugatedHessianAmplitude);

			m_Project.SetShrinkedHessian(t, m_CurrentLevel, shrinkedHessian);
			//WriteFile<HessianImageType>(std::string(directory),std::string("shrinkedHessian"),frameNum.str(),"mha",shrinkedHessian);

			conjugatedHessianAmplitude->SetSpacing(originalSpacing);
			conjugatedHessianAmplitude->SetOrigin(originalOrigin);

			m_Project.SetConjugatedHessian(t, m_CurrentLevel, conjugatedHessian);
		}

		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {

			//		typename InputImageType::Pointer currentEstimate = ReadFile<InputImageType>(std::string(directory),std::string("estimate"),frameNum.str(),"mha");

			typename InputImageType::Pointer boundsLagrange { };
			boundsLagrange = m_Project.GetBoundsLagrange(t, m_CurrentLevel);

			//ReadFile<InputImageType>(std::string(directory),std::string("boundsLagrange"),frameNum.str(),"mha",boundsLagrange);

			typename InputImageType::Pointer bounded { };
			typename InputImageType::Pointer conjugatedBounded { };

			typename InputImageType::Pointer estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);
			DoBoundsStage<InputImageType, ComplexImageType>(estimate, boundsLagrange, m_PadSize, m_PadLowerBound, bounded, conjugatedBounded);

			m_Project.SetShrinkedBounded(t, m_CurrentLevel, bounded);
			//WriteFile<InputImageType>(std::string(directory),std::string("bounded"),frameNum.str(),"mha",bounded);

			m_Project.SetConjugatedBounded(t, m_CurrentLevel, conjugatedBounded);

			//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedBounded"),frameNum.str(),"mha",conjugatedBounded);

			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {
				if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
					continue;
				typedef itk::ContinuousBorderWarpImageFilter<InputImageType, InputImageType, FieldType> MovingImageWarperType;

				typename InputImageType::Pointer movingEstimate, shrinkedMoving, conjugatedMoving, lagrangeMoving;

				typename MovingImageWarperType::Pointer warper = MovingImageWarperType::New();

				typename FieldType::Pointer registrationField;
				//registrationField = m_Project.GetMotionField(t1, t, m_CurrentLevel);
				registrationField = m_Project.GetMotionField(t, t1, m_CurrentLevel);

				warper->SetInput(estimate);
				warper->SetOutputParametersFromImage(estimate);
				warper->SetDisplacementField(registrationField);
				warper->Update();

				shrinkedMoving = warper->GetOutput();
				shrinkedMoving->DisconnectPipeline();

				m_Project.SetMovingShrinked(t1, t, m_CurrentLevel, shrinkedMoving);

				typedef itk::SubtractImageFilter<InputImageType, InputImageType, InputImageType> SubtractImageFilterType;

				typename SubtractImageFilterType::Pointer subtractor = SubtractImageFilterType::New();
				lagrangeMoving = m_Project.GetMovingLagrange(t1, t, m_CurrentLevel);
				lagrangeMoving->SetOrigin(shrinkedMoving->GetOrigin());
				subtractor->SetInput1(shrinkedMoving);
				subtractor->SetInput2(lagrangeMoving);

				subtractor->Update();

				conjugatedMoving = subtractor->GetOutput();
				m_Project.SetMovingConjugated(t1, t, m_CurrentLevel, conjugatedMoving);

			}

		}						//DO SHRINK


	}
	void DoReconstructStep() {
		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {
			std::cout << "RECONSTRUCT T " << t << std::endl;
			typename InputImageType::Pointer conjugatedPoisson, paddedConjugatedPoisson, conjugatedHessian, paddedConjugatedHessian,
					conjugatedBounded, paddedConjugatedBounded, totalPadded, total, normalizer;

			typename ComplexImageType::Pointer conjugatedPoissonFrequency, conjugatedHessianFrequency, conjugatedBoundedFrequency, transfer,
					totalFrequency;

			typedef itk::AddImageFilter<ComplexImageType, ComplexImageType, ComplexImageType> ComplexAccumulator;
			typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> Accumulator;

			typename ComplexAccumulator::Pointer complexAccumulator = ComplexAccumulator::New();
			typename Accumulator::Pointer normalizerAccumulator = Accumulator::New();

			//UPDATES
			conjugatedPoisson = m_Project.GetConjugatedPoisson(t, m_CurrentLevel);
			PadImage<InputImageType>(conjugatedPoisson, m_PadSize, m_PadLowerBound, paddedConjugatedPoisson);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedPoisson, conjugatedPoissonFrequency);

			conjugatedHessianFrequency = m_Project.GetConjugatedHessian(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(conjugatedHessianFrequency, m_Offset, conjugatedHessianFrequency);
			conjugatedHessianFrequency->SetOrigin(conjugatedPoissonFrequency->GetOrigin());
#if 0
			PadImage<InputImageType>(conjugatedHessian, padSize, padLowerBound, paddedConjugatedHessian);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedHessian, conjugatedHessianFrequency);
#endif
			conjugatedBounded = m_Project.GetConjugatedBounded(t, m_CurrentLevel);
			PadImage<InputImageType>(conjugatedBounded, m_PadSize, m_PadLowerBound, paddedConjugatedBounded);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedBounded, conjugatedBoundedFrequency);

			complexAccumulator->SetInput1(conjugatedPoissonFrequency);
			complexAccumulator->SetInput2(conjugatedHessianFrequency);
			complexAccumulator->Update();

			totalFrequency = complexAccumulator->GetOutput();
			totalFrequency->DisconnectPipeline();

			complexAccumulator->SetInput1(totalFrequency);
			complexAccumulator->SetInput2(conjugatedBoundedFrequency);
			complexAccumulator->Update();

			totalFrequency = complexAccumulator->GetOutput();
			totalFrequency->DisconnectPipeline();

			//NORMALIZERS

			//POISSON
			std::cout << "Poisson" << std::endl;
			transfer = m_Project.GetTransferImage(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);

			typedef itk::ComplexToModulusImageAdaptor<ComplexImageType, double> ModulusFilter;
			typename ModulusFilter::Pointer modulusAdaptor = ModulusFilter::New();
			modulusAdaptor->SetImage(transfer);

			typedef itk::SquareImageFilter<ModulusFilter, InputImageType> SquareFilter;
			typename SquareFilter::Pointer squareFilter = SquareFilter::New();
			squareFilter->SetInput(modulusAdaptor);
			squareFilter->Update();

			normalizer = squareFilter->GetOutput();
			normalizer->DisconnectPipeline();

			//HESSIAN
			std::cout << "Hessian" << std::endl;
			normalizerAccumulator->SetInput1(normalizer);
			m_HessianNormalizer->SetOrigin(normalizer->GetOrigin());
			normalizerAccumulator->SetInput2(m_HessianNormalizer);
			normalizerAccumulator->Update();

			normalizer = normalizerAccumulator->GetOutput();
			normalizer->DisconnectPipeline();

			//BOUNDS
			std::cout << "Bounds" << std::endl;
			typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
			typename AddType::Pointer addScalarFilter = AddType::New();
			addScalarFilter->SetInput1(normalizer);
			addScalarFilter->SetConstant2(1);
			addScalarFilter->Update();

			normalizer = addScalarFilter->GetOutput();
			normalizer->DisconnectPipeline();

			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {

				if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
					continue;
				{
#if 1
				typename InputImageType::Pointer conjugatedMovingFrame, paddedConjugatedMovingFrame;
				typename ComplexImageType::Pointer conjugatedMovingFrameFrequency;

				//conjugatedMovingFrame = m_Project.GetMovingConjugated(t1, t, m_CurrentLevel);
				conjugatedMovingFrame = m_Project.GetMovingConjugated(t, t1, m_CurrentLevel);

				PadImage<InputImageType>(conjugatedMovingFrame, m_PadSize, m_PadLowerBound, paddedConjugatedMovingFrame);
				ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedMovingFrame, conjugatedMovingFrameFrequency);

				complexAccumulator->SetInput1(totalFrequency);
				complexAccumulator->SetInput2(conjugatedMovingFrameFrequency);
				complexAccumulator->Update();

				totalFrequency = complexAccumulator->GetOutput();
				totalFrequency->DisconnectPipeline();

				normalizerAccumulator->SetInput1(normalizer);
				normalizerAccumulator->SetInput2(squareFilter->GetOutput());
				normalizerAccumulator->Update();

				normalizer = normalizerAccumulator->GetOutput();
				normalizer->DisconnectPipeline();

				typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
				typename AddType::Pointer addScalarFilter = AddType::New();
				addScalarFilter->SetInput1(normalizer);
				addScalarFilter->SetConstant2(1);
				addScalarFilter->Update();

				normalizer = addScalarFilter->GetOutput();
				normalizer->DisconnectPipeline();
#endif
				}

#if 0
				typename InputImageType::Pointer conjugatedWarpedPoissonFrame, paddedConjugatedWarpedPoissonFrame;
				typename ComplexImageType::Pointer conjugatedWarpedPoisonFrameFrequency;

				conjugatedWarpedPoissonFrame=m_Project.GetConjugatedWarpedPoisson(t1,t,m_CurrentLevel);


				PadImage<InputImageType>(conjugatedWarpedPoissonFrame, m_PadSize, m_PadLowerBound, paddedConjugatedWarpedPoissonFrame);
				ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedWarpedPoissonFrame, conjugatedWarpedPoisonFrameFrequency);

				complexAccumulator->SetInput1(totalFrequency);
				complexAccumulator->SetInput2(conjugatedWarpedPoisonFrameFrequency);
				complexAccumulator->Update();

				totalFrequency = complexAccumulator->GetOutput();
				totalFrequency->DisconnectPipeline();


				normalizerAccumulator->SetInput1(normalizer);
				m_HessianNormalizer->SetOrigin(normalizer->GetOrigin());
				normalizerAccumulator->SetInput2(squareFilter->GetOutput());
				normalizerAccumulator->Update();

				normalizer = normalizerAccumulator->GetOutput();
				normalizer->DisconnectPipeline();
#endif
			}

			typedef itk::DivideImageFilter<ComplexImageType, InputImageType, ComplexImageType> ComplexDividerType;

			typename ComplexDividerType::Pointer complexDivider = ComplexDividerType::New();
			totalFrequency->SetOrigin(normalizer->GetOrigin());
			complexDivider->SetInput1(totalFrequency);
			complexDivider->SetInput2(normalizer);
			complexDivider->Update();

			totalFrequency = complexDivider->GetOutput();
			totalFrequency->DisconnectPipeline();

			m_Project.SetEstimatedFrequencyImage(t, m_CurrentLevel, totalFrequency);

			ImageIFFT<InputImageType, ComplexImageType>(totalFrequency, totalPadded);
			CropImage<InputImageType>(totalPadded, conjugatedPoisson->GetLargestPossibleRegion(), total);

			auto origin = total->GetOrigin();
			origin.Fill(0);
			total->SetOrigin(origin);

			total->SetSpacing(conjugatedPoisson->GetSpacing());
			m_Project.SetEstimatedImage(t, m_CurrentLevel, total);

		}

	}
	void DoLagrangeStep(){


		typedef itk::ContinuousBorderWarpImageFilter<InputImageType, InputImageType, FieldType> MovingImageWarperType;


		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;

			typename ComplexImageType::Pointer resultFrequency, transfer;
			typename InputImageType::Pointer result, poissonLagrange, denoised;
			//std::stringstream frameNum("");
			//frameNum << "_T"<< t;

			result = m_Project.GetEstimatedImage(t, m_CurrentLevel);
			//ReadFile<InputImageType>(std::string(directory),std::string("estimate"),frameNum.str(),"mha",result);

			resultFrequency = m_Project.GetEstimatedFrequencyImage(t, m_CurrentLevel);
			//ReadFile<ComplexImageType>(std::string(directory),std::string("estimateFrequency"),frameNum.str(),"mha",resultFrequency);
			{

				transfer = m_Project.GetTransferImage(t, m_CurrentLevel);
				//ReadFile<ComplexImageType>(std::string(directory),std::string("transfer"),frameNum.str(),"mha",transfer);

				poissonLagrange = m_Project.GetPoissonLagrange(t, m_CurrentLevel);
				//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);
				denoised = m_Project.GetPoissonShrinkedImage(t, m_CurrentLevel);
				//ReadFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);

				AdjustImage<ComplexImageType>(resultFrequency, m_Offset, resultFrequency);
				AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);

				auto origin = resultFrequency->GetOrigin();
				origin.Fill(0);

				resultFrequency->SetOrigin(origin);
				transfer->SetOrigin(origin);
				denoised->SetOrigin(origin);

				resultFrequency->SetSpacing(result->GetSpacing());
				transfer->SetSpacing(result->GetSpacing());

				typedef itk::MultiplyImageFilter<ComplexImageType, ComplexImageType, ComplexImageType> ComplexMultiplyType;

				typename ComplexMultiplyType::Pointer complexMultiplyFilter4 = ComplexMultiplyType::New();
				complexMultiplyFilter4->SetInput1(resultFrequency);
				complexMultiplyFilter4->SetInput2(transfer);

				typedef itk::HalfHermitianToRealInverseFFTImageFilter<ComplexImageType, InputImageType> IFFTFilterType;
				typename IFFTFilterType::Pointer IFFTFilter3 = IFFTFilterType::New();
				IFFTFilter3->SetInput(complexMultiplyFilter4->GetOutput());
				IFFTFilter3->Update();
				typename InputImageType::Pointer tmp = IFFTFilter3->GetOutput();

				CropImage<InputImageType>(tmp, result->GetLargestPossibleRegion(), tmp);

				origin.Fill(0);
				tmp->SetOrigin(origin);
				poissonLagrange->SetOrigin(origin);
				tmp->SetSpacing(result->GetSpacing());

				poissonLagrange->SetSpacing(result->GetSpacing());

				typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
				typename AddType::Pointer addFilter3 = AddType::New();
				addFilter3->SetInput1(tmp);
				addFilter3->SetInput2(poissonLagrange);

				typedef itk::SubtractImageFilter<InputImageType, InputImageType, InputImageType> SubType;
				typename SubType::Pointer subFilter3 = SubType::New();
				subFilter3->SetInput1(addFilter3->GetOutput());
				subFilter3->SetInput2(denoised);
				subFilter3->Update();
				poissonLagrange = subFilter3->GetOutput();
				poissonLagrange->DisconnectPipeline();

				m_Project.SetPoissonLagrange(t, m_CurrentLevel, poissonLagrange);
				//WriteFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);



			}
#if 0
			{

				for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {
					if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
						continue;

					auto registrationField = m_Project.GetMotionField(t, t1, m_CurrentLevel);

					typename MovingImageWarperType::Pointer warper = MovingImageWarperType::New();

					warper->SetInput(result);
					warper->SetOutputParametersFromImage(result);
					warper->SetDisplacementField(registrationField);
					warper->Update();
					typename InputImageType::Pointer warpedResult=warper->GetOutput();
					warpedResult->DisconnectPipeline();

					typename ComplexImageType::Pointer warpedtransfer = m_Project.GetTransferImage(t1, m_CurrentLevel);
									//ReadFile<ComplexImageType>(std::string(directory),std::string("transfer"),frameNum.str(),"mha",transfer);

					typename InputImageType::Pointer warpedPoissonLagrange = m_Project.GetWarpedPoissonLagrange(t,t1, m_CurrentLevel);
					//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);
					auto warpedDenoised = m_Project.GetWarpedPoissonShrinkedImage(t,t1, m_CurrentLevel);
									//ReadFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);


					typename InputImageType::Pointer paddedWarpedResult;
					typename ComplexImageType::Pointer warpedResultFrequency;
					PadImage<InputImageType>(warpedResult, m_PadSize, m_PadLowerBound, paddedWarpedResult);
					ImageFFT<InputImageType, ComplexImageType>(paddedWarpedResult, warpedResultFrequency);


					//AdjustImage<ComplexImageType>(warpedResultFrequency, m_Offset, warpedResultFrequency);

					AdjustImage<ComplexImageType>(warpedtransfer, m_Offset, warpedtransfer);

					warpedResultFrequency->GetOrigin().Fill(0);
					warpedtransfer->GetOrigin().Fill(0);
					warpedDenoised->GetOrigin().Fill(0);

					warpedResultFrequency->SetSpacing(result->GetSpacing());
					warpedtransfer->SetSpacing(result->GetSpacing());

					typedef itk::MultiplyImageFilter<ComplexImageType, ComplexImageType, ComplexImageType> ComplexMultiplyType;

					typename ComplexMultiplyType::Pointer complexMultiplyFilter4 = ComplexMultiplyType::New();
					complexMultiplyFilter4->SetInput1(warpedResultFrequency);
					complexMultiplyFilter4->SetInput2(warpedtransfer);

					typedef itk::HalfHermitianToRealInverseFFTImageFilter<ComplexImageType, InputImageType> IFFTFilterType;
					typename IFFTFilterType::Pointer IFFTFilter3 = IFFTFilterType::New();
					IFFTFilter3->SetInput(complexMultiplyFilter4->GetOutput());
					IFFTFilter3->Update();
					typename InputImageType::Pointer tmp = IFFTFilter3->GetOutput();

					CropImage<InputImageType>(tmp, result->GetLargestPossibleRegion(), tmp);
					tmp->GetOrigin().Fill(0);
					warpedPoissonLagrange->GetOrigin().Fill(0);
					tmp->SetSpacing(result->GetSpacing());
					warpedPoissonLagrange->SetSpacing(result->GetSpacing());

					typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
					typename AddType::Pointer addFilter3 = AddType::New();
					addFilter3->SetInput1(tmp);
					addFilter3->SetInput2(warpedPoissonLagrange);

					typedef itk::SubtractImageFilter<InputImageType, InputImageType, InputImageType> SubType;
					typename SubType::Pointer subFilter3 = SubType::New();
					subFilter3->SetInput1(addFilter3->GetOutput());
					subFilter3->SetInput2(warpedDenoised);
					subFilter3->Update();
					warpedPoissonLagrange = subFilter3->GetOutput();
					warpedPoissonLagrange->DisconnectPipeline();

					this->m_Project.SetWarpedPoissonLagrange(t,t1,m_CurrentLevel,warpedPoissonLagrange);


				}
			}
#endif
			//////////////////
			{
				typename HessianImageType::Pointer hessianLagrange, shrinkedHessian;
				hessianLagrange = m_Project.GetHessianLagrange(t, m_CurrentLevel);
				hessianLagrange->SetOrigin(this->m_ZeroOrigin);
				//ReadFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);
				shrinkedHessian = m_Project.GetShrinkedHessian(t, m_CurrentLevel);
				//ReadFile<HessianImageType>(std::string(directory),std::string("shrinkedHessian"),frameNum.str(),"mha",shrinkedHessian);

				typedef ttt::MultiplyByTensorImageFilter<ComplexImageType, HessianComplexImageType> HessianFrequencyFilterType;

				typedef ttt::HessianIFFTImageFilter<HessianComplexImageType, HessianImageType> HessianIFFTFilterType;
				typedef itk::ExtractImageFilter<HessianImageType, HessianImageType> ExtractFilterType;
				typedef ttt::HessianFFTImageFilter<HessianImageType, HessianComplexImageType> HessianFFTFilterType;

				typedef itk::AddImageFilter<HessianImageType> AddHessianType;
				typedef itk::SubtractImageFilter<HessianImageType> SubHessianType;

				typename HessianFrequencyFilterType::Pointer hessianFrequency = HessianFrequencyFilterType::New();
				auto spacing = resultFrequency->GetSpacing();
				spacing.Fill(1);
				resultFrequency->SetSpacing(spacing);
				hessianFrequency->SetInput1(resultFrequency);
				hessianFrequency->SetInput2(m_HessianFilter);

				typename HessianIFFTFilterType::Pointer hessianIFFT = HessianIFFTFilterType::New();
				hessianIFFT->SetInput(hessianFrequency->GetOutput());
				hessianIFFT->Update();
				typename HessianImageType::Pointer hessian = hessianIFFT->GetOutput();
				CropImage<HessianImageType>(hessian, result->GetLargestPossibleRegion(), hessian);

				hessian->SetSpacing(result->GetSpacing());
				hessian->SetOrigin(result->GetOrigin());
				//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedHessian"),frameNum.str(),"mha",conjugatedHessian);
				//WriteFile<HessianImageType>(std::string(directory),std::string("hessian"),frameNum.str(),"mha",hessian);
				m_Project.SetHessian(t, m_CurrentLevel, hessian);

#if 1
				typedef itk::UnaryFunctorImageFilter<HessianImageType, InputImageType, PlatenessFunctor<typename HessianImageType::PixelType> > PlatenessImageFilterType;

				typename PlatenessImageFilterType::Pointer platenessFilter = PlatenessImageFilterType::New();

				platenessFilter->SetInput(hessian);
				platenessFilter->Update();
				typename InputImageType::Pointer plateness = platenessFilter->GetOutput();
				plateness->DisconnectPipeline();

				//WriteFile<InputImageType>(std::string(m_Directory), std::string("plateness"), frameNum.str(), "mha", plateness);
#endif

				typename AddHessianType::Pointer adder = AddHessianType::New();
				adder->SetInput1(hessianLagrange);

				adder->SetInput2(hessian);

				typename SubHessianType::Pointer subtractor = SubHessianType::New();
				subtractor->SetInput1(adder->GetOutput());
				shrinkedHessian->SetSpacing(result->GetSpacing());
				subtractor->SetInput2(shrinkedHessian);
				subtractor->Update();
				hessianLagrange = subtractor->GetOutput();
				hessianLagrange->DisconnectPipeline();

				m_Project.SetHessianLagrange(t, m_CurrentLevel, hessianLagrange);
				//WriteFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);
			}

			typedef itk::AddImageFilter<InputImageType, InputImageType> AddType;
			typedef itk::SubtractImageFilter<InputImageType, InputImageType> SubType;

			typename InputImageType::Pointer boundsLagrange { };
			boundsLagrange = m_Project.GetBoundsLagrange(t, m_CurrentLevel);
			auto origin = boundsLagrange->GetOrigin();
			origin.Fill(0);
			boundsLagrange->SetOrigin(origin);
			//ReadFile<InputImageType>(std::string(directory),std::string("boundsLagrange"),frameNum.str(),"mha",boundsLagrange);

			typename InputImageType::Pointer bounded { };
			bounded = m_Project.GetShrinkedBounded(t, m_CurrentLevel);
			bounded->SetOrigin(origin);
			//ReadFile<InputImageType>(std::string(directory),std::string("bounded"),frameNum.str(),"mha",bounded);

			typename AddType::Pointer adder = AddType::New();
			adder->SetInput1(result);
			adder->SetInput2(boundsLagrange);
			typename SubType::Pointer subtractor = SubType::New();

			subtractor->SetInput1(adder->GetOutput());
			subtractor->SetInput2(bounded);
			subtractor->Update();
			boundsLagrange = subtractor->GetOutput();
			boundsLagrange->DisconnectPipeline();
			m_Project.SetBoundsLagrange(t, m_CurrentLevel, boundsLagrange);

			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {

				if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
					continue;

				typename InputImageType::Pointer shrinkedMoving, lagrangeMoving;

				shrinkedMoving = m_Project.GetMovingShrinked(t, t1, m_CurrentLevel);
				lagrangeMoving = m_Project.GetMovingLagrange(t, t1, m_CurrentLevel);

				auto origin = shrinkedMoving->GetOrigin();
				origin.Fill(0);
				shrinkedMoving->SetOrigin(origin);
				lagrangeMoving->SetOrigin(origin);

				typename AddType::Pointer adder = AddType::New();
				adder->SetInput1(result);
				adder->SetInput2(lagrangeMoving);

				typename SubType::Pointer subtractor = SubType::New();

				subtractor->SetInput1(adder->GetOutput());
				subtractor->SetInput2(shrinkedMoving);
				subtractor->Update();
				lagrangeMoving = subtractor->GetOutput();
				lagrangeMoving->DisconnectPipeline();

				m_Project.SetMovingLagrange(t, t1, m_CurrentLevel, lagrangeMoving);

			}

		}

	}
	virtual void DeconvIter(){
			this->DoShrinkStep();
			this->DoReconstructStep();
			this->DoLagrangeStep();
		}
	virtual void DoBlind(){
		for(int it=0;it< this->m_NumberOfBlindIterations;it++){
		for (int t = 0; t <m_Project.GetNumberOfFrames(); t++) {

			typename ComplexImageType::Pointer imageEstimateFrequency, transferCurrentEstimateFrequency, transferNextEstimateFrequency;
			typename InputImageType::Pointer original, paddedOriginal, estimatedPSF, image, paddedImage;
			original = m_Project.GetObservedImage(t, m_CurrentLevel);
			image = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			transferCurrentEstimateFrequency = m_Project.GetTransferImage(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(transferCurrentEstimateFrequency, m_Offset, transferCurrentEstimateFrequency);

			typename InputImageType::SpacingType oneSpacing = original->GetSpacing();
			typename InputImageType::PointType zeroOrigin = original->GetOrigin();

			oneSpacing.Fill(1);
			zeroOrigin.Fill(0);

			original->SetSpacing(oneSpacing);

			transferCurrentEstimateFrequency->SetSpacing(oneSpacing);


			transferCurrentEstimateFrequency->SetOrigin(zeroOrigin);

			original->SetOrigin(zeroOrigin);

			PadImage<InputImageType>(image, m_PadSize, m_PadLowerBound, paddedImage);

			typedef itk::MaximumImageFilter<InputImageType> MaximumType;

			typename MaximumType::Pointer maximum = MaximumType::New();
			maximum->SetInput1(paddedImage);
			maximum->SetConstant2(0.0);
			maximum->Update();
			ImageFFT<InputImageType, ComplexImageType>(maximum->GetOutput(), imageEstimateFrequency);

			PadImage<InputImageType>(original, m_PadSize, m_PadLowerBound, paddedOriginal);

			BlindRL<InputImageType, ComplexImageType>(transferCurrentEstimateFrequency, imageEstimateFrequency, paddedOriginal, estimatedPSF,
					transferNextEstimateFrequency);
			//WriteFile<InputImageType>(std::string(m_Directory), std::string("estimatedPSF"), frameNum.str(), "mha", estimatedPSF);
			m_Project.SetPSF(t,m_CurrentLevel,estimatedPSF);
			m_Project.SetTransferImage(t, m_CurrentLevel, transferNextEstimateFrequency);

		} //DO BLIND
		}
	}
	virtual void DoUpsample() {

		int nextLevel = m_CurrentLevel + 1;
		typename InputImageType::SpacingType nextSpacing = m_Project.GetObservedImage(0, nextLevel)->GetSpacing();
		typename InputImageType::SizeType nextSize = m_Project.GetObservedImage(0, nextLevel)->GetLargestPossibleRegion().GetSize();

		for (int t = 0; t < m_Project.GetNumberOfFrames(); t++) {

			typedef itk::ResampleImageFilter<InputImageType, InputImageType> ResampleFilterType;
			typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();
			typedef itk::LinearInterpolateImageFunction<InputImageType, double> Interpolator;
			typedef itk::NearestNeighborExtrapolateImageFunction<InputImageType, double> Extrapolator;
			typename Interpolator::Pointer interpolator = Interpolator::New();
			typename Extrapolator::Pointer extrapolator = Extrapolator::New();

			resampler->SetInterpolator(interpolator);
			resampler->SetExtrapolator(extrapolator);
			resampler->SetInput(m_Project.GetEstimatedImage(t, m_CurrentLevel));
			resampler->SetSize(nextSize);
			resampler->SetOutputSpacing(nextSpacing);
			resampler->Update();

			m_Project.SetEstimatedImage(t, nextLevel, resampler->GetOutput());

			typedef itk::VectorResampleImageFilter<FieldType, FieldType> FieldExpanderType;

			typename FieldExpanderType::Pointer fieldExpander = FieldExpanderType::New();
			fieldExpander->SetSize(nextSize);
			fieldExpander->SetOutputSpacing(nextSpacing);

			for (int t1 = int(t - m_W); t1 <= int(t + m_W); t1++) {
				if (t1 < 0 || t1 >= m_Project.GetNumberOfFrames() || t1 == t)
					continue;

				typename FieldType::Pointer registrationField, nextField;
				registrationField = m_Project.GetMotionField(t, t1, m_CurrentLevel);

				fieldExpander->SetInput(registrationField);
				fieldExpander->UpdateLargestPossibleRegion();

				nextField = fieldExpander->GetOutput();
				nextField->DisconnectPipeline();
				m_Project.SetMotionField(t, t1, nextLevel, nextField);
			}

		} //UPSAMPLE
	}
	virtual void DeconvLoop(){

		for(int it=0;it<m_NumberOfDeconvolutionIterations;it++){
			this->DeconvIter();
		}

	}
	void Do(){
		this->InitPyramid();

		for(m_CurrentLevel=0;m_CurrentLevel<m_NumberOfLevels;m_CurrentLevel++){
			m_FirstRegistration=true;
			this->InitLevelIteration();

			this->InitDeconvolutionIteration();

			//this->m_NumberOfIterations=m_CurrentLevel+1;
			//this->m_NumberOfIterations=10;
			this->DoBlind();
			this->DoRegistration();
			for(m_CurrentIteration=0;m_CurrentIteration< m_NumberOfIterations;m_CurrentIteration++){

				this->DoShrinkStep();
				this->DoReconstructStep();
				this->DoBlind();
				this->DoRegistration();
				this->DoLagrangeStep();
				//this->DeconvLoop();
				//exit(-1);

			}
			if(m_CurrentLevel<m_NumberOfLevels-1){
				this->DoUpsample();
			}
			this->m_NumberOfIterations*=2;
			//this->m_NumberOfIterations*=2;
		}
	}
protected:

private:
	TProject & m_Project;
	unsigned int m_NumberOfLevels;
	unsigned int m_NumberOfIterations;

	unsigned int m_CurrentLevel;
	unsigned int m_CurrentIteration;



	bool m_Restart;
	bool m_FirstRegistration;


	unsigned int m_Factors[3];
	unsigned int m_W;


	unsigned int m_NumberOfDeconvolutionIterations;

	unsigned int m_NumberOfBlindIterations;

	typename HessianComplexImageType::Pointer m_HessianFilter;
	typename InputImageType::Pointer m_HessianNormalizer;

	typename InputImageType::SizeType m_PadSize;
	typename InputImageType::SizeType m_PadLowerBound;

	typename InputImageType::OffsetType m_Offset;

	typename InputImageType::PointType m_ZeroOrigin;
	typename InputImageType::PointType m_OriginalOrigin;

	typename InputImageType::SpacingType m_OneSpacing;
	typename InputImageType::SpacingType m_OriginalSpacing;

};

int main(int argc,char **argv){

	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <directory> <prefix>  <first> <last>" << std::endl;
	}

	unsigned int MaxIterations = 3;
	unsigned int MaxOuterIterations = 20;
	char * directory = argv[1];
	char * prefix = argv[2];
	int first = atoi(argv[3]);
	int last = atoi(argv[4]);

	typedef itk::Image<double,3> FrameType;
	typedef itk::VideoStream<FrameType> InputVideoStream;
	typedef itk::VideoStream<FrameType> OutputVideoStream;

	TrackingAndDeconvolutionProject project { };
	project.NewProject(first, last, std::string(directory),std::string(prefix));

	typedef HSPVRRAL<TrackingAndDeconvolutionProject> DeconvoluterType;

	DeconvoluterType deconvoluter(project);
	deconvoluter.Do();


}

#if 0
int main(int argc,char **argv){

	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <directory> <prefix>  <first> <last>" << std::endl;
	}

	unsigned int MaxIterations = 3;
	unsigned int MaxOuterIterations = 20;
	char * directory = argv[1];
	char * prefix = argv[2];
	int first = atoi(argv[3]);
	int last = atoi(argv[4]);

	typedef itk::Image<double,3> FrameType;
	typedef itk::VideoStream<FrameType> InputVideoStream;
	typedef itk::VideoStream<FrameType> OutputVideoStream;

	typedef HSPVRRAL<InputVideoStream,OutputVideoStream> DeconvoluterType;



	typedef itk::VideoFileReader<InputVideoStream> InputVideoStreamReader;
	typedef itk::VideoFileWriter<OutputVideoStream> OutputVideoStreamWriter;

	typename InputVideoStreamReader::Pointer inputVideoStreamReader = InputVideoStreamReader::New();

	MyDirectoryVideoIO::Pointer inputVideoIO = MyDirectoryVideoIO::New();
	inputVideoStreamReader->SetVideoIO(inputVideoIO);

	typename DeconvoluterType::Pointer deconvoluter=DeconvoluterType::New();
	deconvoluter->SetInput(inputVideoStreamReader->GetOutput());

	typename OutputVideoStreamWriter::Pointer outputVideoStreamWriter = OutputVideoStreamWriter::New();
	outputVideoStreamWriter->SetFileName("foo");
	outputVideoStreamWriter->SetInput(deconvoluter->GetOutput());
	outputVideoStreamWriter->Update();



}

#endif


#ifdef OLD_DECONVOLUTER





//#define ALPHAHESSIAN pow(2.0,4)

#if 1
template<class T> void ReadFile(const std::string & directory, const std::string & prefix, const std::string & sufix, const std::string & fileType,
		typename T::Pointer & result) {

	std::stringstream buffer { };
	buffer << directory << "/" << prefix << sufix << "." << fileType;
	typedef itk::ImageFileReader<T> ReaderType;

	typename ReaderType::Pointer reader = ReaderType::New();
#if 0
	if (fileType == "ome.tif") {
		itk::SCIFIOImageIO::Pointer imageIO = itk::SCIFIOImageIO::New();
		reader->SetImageIO(imageIO);
	}
#endif
	std::cout << buffer.str() << std::endl;
	reader->SetFileName(buffer.str());
	reader->Update();
	result = reader->GetOutput();
	result->DisconnectPipeline();

}
#endif

template<class T> void WriteFile(const std::string & directory, const std::string & prefix, const std::string & sufix, const std::string & fileType,
		const typename T::Pointer & image) {

	std::stringstream buffer { };
	buffer << directory << "/" << prefix << sufix << "." << fileType;
	typedef itk::ImageFileWriter<T> WriterType;

	typename WriterType::Pointer writer = WriterType::New();
	if (fileType == "ome.tif") {
		itk::SCIFIOImageIO::Pointer imageIO = itk::SCIFIOImageIO::New();
		writer->SetImageIO(imageIO);
	}

	std::cout << buffer.str() << std::endl;
	writer->SetFileName(buffer.str());
	writer->SetInput(image);
	writer->Update();
}

template<class TInputImage> void PadImage(const typename TInputImage::Pointer & frame, const typename TInputImage::SizeType & padSize,
		const typename TInputImage::SizeType & padLowerBound, typename TInputImage::Pointer & padded) { //const typename TInputImage::Pointer & psf,typename TInputImage::Pointer & padded){
	//typename TInputImage::SizeType padSize = GetPadSize<TInputImage>(frame,psf);
	//typename TInputImage::SizeType padLowerBound = GetPadLowerBound<TInputImage>(frame,psf);

	typename TInputImage::SizeType inputSize = frame->GetLargestPossibleRegion().GetSize();
	typedef PadImageFilter<TInputImage, TInputImage> InputPadFilterType;
	typename InputPadFilterType::Pointer inputPadder = InputPadFilterType::New();
	typedef typename itk::ZeroFluxNeumannBoundaryCondition<TInputImage> BoundaryConditionType;

	BoundaryConditionType * boundaryCondition = new BoundaryConditionType { };
	inputPadder->SetBoundaryCondition(boundaryCondition);

	inputPadder->SetPadLowerBound(padLowerBound);

	typename TInputImage::SizeType inputUpperBound { };
	for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
		inputUpperBound[i] = (padSize[i] - inputSize[i]) / 2;
		if ((padSize[i] - inputSize[i]) % 2 == 1) {
			inputUpperBound[i]++;
		}
	}
	inputPadder->SetPadUpperBound(inputUpperBound);
	//inputPadder->SetNumberOfThreads( this->GetNumberOfThreads() );
	inputPadder->SetInput(frame);

	inputPadder->Update();
	padded = inputPadder->GetOutput();
	padded->DisconnectPipeline();

}

template<class TImage> void CropImage(const typename TImage::Pointer & extended, const typename TImage::RegionType & cropRegion,
		typename TImage::Pointer & result) {
	typedef itk::ExtractImageFilter<TImage, TImage> ExtractFilterType;
	typename ExtractFilterType::Pointer extractor = ExtractFilterType::New();
	extractor = ExtractFilterType::New();
	extractor->SetInput(extended);
	extractor->SetExtractionRegion(cropRegion);
	extractor->Update();
	result = extractor->GetOutput();
	result->DisconnectPipeline();

}



#if 0
template< typename TInputImage, typename TKernelImage, typename TOutputImage, typename TInternalPrecision >
void
FFTConvolutionImageFilter< TInputImage, TKernelImage, TOutputImage, TInternalPrecision >
::TransformPaddedInput(const InternalImageType * paddedInput,
		InternalComplexImagePointerType & transformedInput,
		ProgressAccumulator * progress, float progressWeight)
{
	// Take the Fourier transform of the padded image.
	typename FFTFilterType::Pointer imageFFTFilter = FFTFilterType::New();
	imageFFTFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	imageFFTFilter->SetInput( paddedInput );
	imageFFTFilter->ReleaseDataFlagOn();
	progress->RegisterInternalFilter( imageFFTFilter, progressWeight );
	imageFFTFilter->Update();

	transformedInput = imageFFTFilter->GetOutput();
	transformedInput->DisconnectPipeline();

	imageFFTFilter->SetInput( ITK_NULLPTR );
	imageFFTFilter = ITK_NULLPTR;
}

template< typename TInputImage, typename TKernelImage, typename TOutputImage, typename TInternalPrecision >
void
FFTConvolutionImageFilter< TInputImage, TKernelImage, TOutputImage, TInternalPrecision >
::PrepareKernel(const KernelImageType * kernel,
		InternalComplexImagePointerType & preparedKernel,
		ProgressAccumulator * progress, float progressWeight)
{
	KernelRegionType kernelRegion = kernel->GetLargestPossibleRegion();
	KernelSizeType kernelSize = kernelRegion.GetSize();

	InputSizeType padSize = this->GetPadSize();
	typename KernelImageType::SizeType kernelUpperBound;
	for (unsigned int i = 0; i < ImageDimension; ++i)
	{
		kernelUpperBound[i] = padSize[i] - kernelSize[i];
	}

	InternalImagePointerType paddedKernelImage = ITK_NULLPTR;

	float paddingWeight = 0.2f;
	if ( this->GetNormalize() )
	{
		typedef NormalizeToConstantImageFilter< KernelImageType, InternalImageType >
		NormalizeFilterType;
		typename NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
		normalizeFilter->SetConstant( NumericTraits< TInternalPrecision >::OneValue() );
		normalizeFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
		normalizeFilter->SetInput( kernel );
		normalizeFilter->ReleaseDataFlagOn();
		progress->RegisterInternalFilter( normalizeFilter,
				0.2f * paddingWeight * progressWeight );

		// Pad the kernel image with zeros.
		typedef ConstantPadImageFilter< InternalImageType, InternalImageType > KernelPadType;
		typedef typename KernelPadType::Pointer KernelPadPointer;
		KernelPadPointer kernelPadder = KernelPadType::New();
		kernelPadder->SetConstant( NumericTraits< TInternalPrecision >::ZeroValue() );
		kernelPadder->SetPadUpperBound( kernelUpperBound );
		kernelPadder->SetNumberOfThreads( this->GetNumberOfThreads() );
		kernelPadder->SetInput( normalizeFilter->GetOutput() );
		kernelPadder->ReleaseDataFlagOn();
		progress->RegisterInternalFilter( kernelPadder,
				0.8f * paddingWeight * progressWeight );
		paddedKernelImage = kernelPadder->GetOutput();
	}
	else
	{
		// Pad the kernel image with zeros.
		typedef ConstantPadImageFilter< KernelImageType, InternalImageType > KernelPadType;
		typedef typename KernelPadType::Pointer KernelPadPointer;
		KernelPadPointer kernelPadder = KernelPadType::New();
		kernelPadder->SetConstant( NumericTraits< TInternalPrecision >::ZeroValue() );
		kernelPadder->SetPadUpperBound( kernelUpperBound );
		kernelPadder->SetNumberOfThreads( this->GetNumberOfThreads() );
		kernelPadder->SetInput( kernel );
		kernelPadder->ReleaseDataFlagOn();
		progress->RegisterInternalFilter( kernelPadder,
				paddingWeight * progressWeight );
		paddedKernelImage = kernelPadder->GetOutput();
	}

	// Shift the padded kernel image.
	typedef CyclicShiftImageFilter< InternalImageType, InternalImageType > KernelShiftFilterType;
	typename KernelShiftFilterType::Pointer kernelShifter = KernelShiftFilterType::New();
	typename KernelShiftFilterType::OffsetType kernelShift;
	for (unsigned int i = 0; i < ImageDimension; ++i)
	{
		kernelShift[i] = -(kernelSize[i] / 2);
	}
	kernelShifter->SetShift( kernelShift );
	kernelShifter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelShifter->SetInput( paddedKernelImage );
	kernelShifter->ReleaseDataFlagOn();
	progress->RegisterInternalFilter( kernelShifter, 0.1f * progressWeight );

	typename FFTFilterType::Pointer kernelFFTFilter = FFTFilterType::New();
	kernelFFTFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelFFTFilter->SetInput( kernelShifter->GetOutput() );
	progress->RegisterInternalFilter( kernelFFTFilter, 0.699f * progressWeight );
	kernelFFTFilter->Update();

	typedef ChangeInformationImageFilter< InternalComplexImageType > InfoFilterType;
	typename InfoFilterType::Pointer kernelInfoFilter = InfoFilterType::New();
	kernelInfoFilter->ChangeRegionOn();

	typedef typename InfoFilterType::OutputImageOffsetValueType InfoOffsetValueType;
	InputSizeType inputLowerBound = this->GetPadLowerBound();
	InputIndexType inputIndex = this->GetInput()->GetLargestPossibleRegion().GetIndex();
	KernelIndexType kernelIndex = kernel->GetLargestPossibleRegion().GetIndex();
	InfoOffsetValueType kernelOffset[ImageDimension];
	for (int i = 0; i < ImageDimension; ++i)
	{
		kernelOffset[i] = static_cast< InfoOffsetValueType >( inputIndex[i] - inputLowerBound[i] - kernelIndex[i] );
	}
	kernelInfoFilter->SetOutputOffset( kernelOffset );
	kernelInfoFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	kernelInfoFilter->SetInput( kernelFFTFilter->GetOutput() );
	progress->RegisterInternalFilter( kernelInfoFilter, 0.001f * progressWeight );
	kernelInfoFilter->Update();

	preparedKernel = kernelInfoFilter->GetOutput();
}
#endif



template<class TInputImage, class TComplexImage> void ImageFFT(const typename TInputImage::Pointer & input,
		typename TComplexImage::Pointer & transformed) {

	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TInputImage, TComplexImage> FFTFilterType;
	// Take the Fourier transform of the padded image.
	typename FFTFilterType::Pointer imageFFTFilter = FFTFilterType::New();
	//imageFFTFilter->SetNumberOfThreads( this->GetNumberOfThreads() );
	imageFFTFilter->SetInput(input);

	imageFFTFilter->Update();

	transformed = imageFFTFilter->GetOutput();
	transformed->DisconnectPipeline();

}


template<class TImage, class TComplexImage> void BlindRL(const typename TComplexImage::Pointer & currentTransferEstimate,
		const typename TComplexImage::Pointer & currentImageEstimate, const typename TImage::Pointer & paddedOriginal, typename TImage::Pointer & psf,
		typename TComplexImage::Pointer & transferNext) {

	typedef itk::MultiplyImageFilter<TComplexImage, TComplexImage, TComplexImage> ComplexMultiplyType;
	// Set up minipipeline to compute estimate at each iteration
	typename ComplexMultiplyType::Pointer complexMultiplyFilter1 = ComplexMultiplyType::New();

	// Transformed estimate will be set as input 1 in Iteration()
	complexMultiplyFilter1->SetInput1(currentImageEstimate);
	complexMultiplyFilter1->SetInput2(currentTransferEstimate);

	typedef itk::HalfHermitianToRealInverseFFTImageFilter<TComplexImage, TImage> IFFTFilterType;
	typename IFFTFilterType::Pointer iFFTFilter1 = IFFTFilterType::New();
	iFFTFilter1->SetInput(complexMultiplyFilter1->GetOutput());

	typedef itk::DivideImageFilter<TImage, TImage, TImage> DivideFilterType;
	typename DivideFilterType::Pointer divideFilter = DivideFilterType::New();

	divideFilter->SetInput1(paddedOriginal);
	divideFilter->SetInput2(iFFTFilter1->GetOutput());

	typedef itk::RealToHalfHermitianForwardFFTImageFilter<TImage, TComplexImage> FFTFilterType;

	typename FFTFilterType::Pointer fFTFilter = FFTFilterType::New();

	fFTFilter->SetInput(divideFilter->GetOutput());

	typedef itk::ComplexConjugateImageAdaptor<TComplexImage> ConjugateAdaptorType;
	typename ConjugateAdaptorType::Pointer conjugateAdaptor = ConjugateAdaptorType::New();
	conjugateAdaptor->SetImage(currentImageEstimate);

	typedef itk::MultiplyImageFilter<TComplexImage, ConjugateAdaptorType, TComplexImage> ComplexConjugateMultiplyType;
	typename ComplexConjugateMultiplyType::Pointer complexMultiplyFilter2 = ComplexConjugateMultiplyType::New();

	complexMultiplyFilter2->SetInput1(fFTFilter->GetOutput());
	complexMultiplyFilter2->SetInput2(conjugateAdaptor);

	typename IFFTFilterType::Pointer iFFTFilter2 = IFFTFilterType::New();

	iFFTFilter2->SetInput(complexMultiplyFilter2->GetOutput());

	typename IFFTFilterType::Pointer iFFTFilter3 = IFFTFilterType::New();

	iFFTFilter3->SetInput(currentTransferEstimate);
#if 0
	typedef itk::MultiplyImageFilter<TImage, TImage, TImage> MultiplyType;

	typename MultiplyType::Pointer multiply = MultiplyType::New();

	multiply->SetInput1(iFFTFilter2->GetOutput());

	multiply->SetInput2(iFFTFilter3->GetOutput());
#endif

	// multiply the result with the input
	typedef itk::BinaryFunctorImageFilter<TImage, TImage, TImage, L1L2Regularization<double> > RegularizerFilterType;

	typename RegularizerFilterType::Pointer regularizer = RegularizerFilterType::New();
	regularizer->SetInput(0, iFFTFilter2->GetOutput());
	regularizer->SetInput(1, iFFTFilter3->GetOutput());
	regularizer->GetFunctor().m_Lambda = ALPHAPSF;
	regularizer->Update();

	typedef itk::NormalizeToConstantImageFilter<TImage, TImage> NormalizeFilterType;
	typename NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
	normalizeFilter->SetConstant(NumericTraits<double>::OneValue());

	normalizeFilter->SetInput(regularizer->GetOutput());
	normalizeFilter->Update();

	psf = normalizeFilter->GetOutput();
	psf->DisconnectPipeline();
#if 0
	typedef itk::MaximumImageFilter<TImage, TImage, TImage> MaximumFilterType;

	typename MaximumFilterType::Pointer maximumFilter = MaximumFilterType::New();
	maximumFilter->SetInput1(multiply->GetOutput());
	maximumFilter->SetConstant2(0.0);

	psf = normalizeFilter->GetOutput();
	psf->DisconnectPipeline();
#endif
	typename FFTFilterType::Pointer fFTFilter2 = FFTFilterType::New();

	fFTFilter2->SetInput(normalizeFilter->GetOutput());

	fFTFilter2->Update();

	transferNext = fFTFilter2->GetOutput();
	transferNext->DisconnectPipeline();

}



class HSPVRRAL2 {

private:
	unsigned int m_First;
	unsigned int m_Last;
	unsigned int m_NumberOfLevels;
	unsigned int m_CurrentLevel;

	unsigned int m_W;

	unsigned int m_MaxDeconvolutionIterations;
	unsigned int m_NumberOfIterations;

	bool m_FirstRegistration;

	std::string m_Directory;
	std::string m_Prefix;

	typedef double FloatType;
	typedef itk::Image<FloatType, 3> InputImageType;
	typedef itk::Image<std::complex<FloatType>, 3> ComplexImageType;
	typedef itk::Image<itk::Vector<FloatType, 3>, 3> FieldType;

	typedef itk::SymmetricSecondRankTensor<FloatType, 3> HessianType;
	typedef itk::Image<HessianType, 3> HessianImageType;
	typedef itk::Image<itk::SymmetricSecondRankTensor<std::complex<FloatType>, 3>, 3> HessianComplexImageType;

	typename HessianComplexImageType::Pointer m_HessianFilter;
	typename InputImageType::Pointer m_HessianNormalizer;

	typename InputImageType::SizeType m_PadSize;
	typename InputImageType::SizeType m_PadLowerBound;

	typename InputImageType::OffsetType m_Offset;

	typename InputImageType::PointType m_ZeroOrigin;
	typename InputImageType::PointType m_OriginalOrigin;

	typename InputImageType::SpacingType m_OneSpacing;
	typename InputImageType::SpacingType m_OriginalSpacing;


private:
	TrackingAndDeconvolutionProject & m_Project;

public:
	HSPVRRAL2(TrackingAndDeconvolutionProject & project){
		m_Project=project;
	}
protected:
	void InitPyramid() {
		typename InputImageType::Pointer psf;

		ReadFile<InputImageType>(std::string(m_Directory), "psf", "", "ome.tif", psf);
		//unsigned int factors[3] = { 8, 8, 2 };
		unsigned int factors[3] = { 32, 32, 5 };
		//unsigned int factors[3] = { 1, 1, 1 };
		typedef itk::MultiResolutionPyramidImageFilter<InputImageType, InputImageType> RecursiveMultiResolutionPyramidImageFilterType;
		typename RecursiveMultiResolutionPyramidImageFilterType::Pointer psfMultiResolutionPyramidImageFilter =
				RecursiveMultiResolutionPyramidImageFilterType::New();
		psfMultiResolutionPyramidImageFilter->SetInput(psf);
		psfMultiResolutionPyramidImageFilter->SetNumberOfLevels(m_NumberOfLevels);
		psfMultiResolutionPyramidImageFilter->SetStartingShrinkFactors(factors);
		psfMultiResolutionPyramidImageFilter->Update();
		for (int t = m_First; t <= m_Last; t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;

			typename InputImageType::Pointer frame;
			ReadFile<InputImageType>(std::string(m_Directory), std::string(m_Prefix), frameNum.str(), "mha", frame);

			typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> ScalerType;
			typename ScalerType::Pointer scaler = ScalerType::New();

			scaler->SetInput1(frame);
			scaler->SetConstant2(255.0);
			//scaler->SetConstant2(4096.0);
			scaler->Update();

			frame = scaler->GetOutput();
			frame->DisconnectPipeline();

			typename RecursiveMultiResolutionPyramidImageFilterType::Pointer recursiveMultiResolutionPyramidImageFilter =
					RecursiveMultiResolutionPyramidImageFilterType::New();
			recursiveMultiResolutionPyramidImageFilter->SetInput(frame);
			recursiveMultiResolutionPyramidImageFilter->SetNumberOfLevels(m_NumberOfLevels);
			recursiveMultiResolutionPyramidImageFilter->SetStartingShrinkFactors(factors);
			recursiveMultiResolutionPyramidImageFilter->Update();

			for (int i = 0; i < m_NumberOfLevels; i++) {
				std::cout << recursiveMultiResolutionPyramidImageFilter->GetOutput(i)->GetLargestPossibleRegion().GetSize() << std::endl;
				std::cout << recursiveMultiResolutionPyramidImageFilter->GetOutput(i)->GetSpacing() << std::endl;
				m_Project.SetOriginalImage(t, i, recursiveMultiResolutionPyramidImageFilter->GetOutput(i));
				m_Project.SetPSF(t, i, psfMultiResolutionPyramidImageFilter->GetOutput(i));
				m_Project.SetEstimatedImage(t, i, recursiveMultiResolutionPyramidImageFilter->GetOutput(i));
			}
		}
	}

	void DoRegistration() {

		for (int t = m_First; t <= m_Last; t++) {

			int moving { t };
			typename InputImageType::Pointer movingFrame { };
			movingFrame = m_Project.GetEstimatedImage(moving, m_CurrentLevel);

			movingFrame->GetOrigin().Fill(0);

			for (unsigned int k = 1; k <= m_W; k++) {
				int fixed { t + k };
				if (fixed < m_First || fixed > m_Last)
					continue;

				typename InputImageType::Pointer fixedFrame { };
				fixedFrame = m_Project.GetEstimatedImage(fixed, m_CurrentLevel);
				//firstFrame = project.GetOriginalImage(fixed, level);
				fixedFrame->GetOrigin().Fill(0);
				//ReadFile<InputImageType>(std::string(directory),std::string("estimate"),firstFrameNum.str(),"mha",firstFrame);
				//typename InputImageType::Pointer firstFrame=ReadFile<InputImageType>(std::string(directory),std::string(prefix),firstFrameNum.str(),"ome.tif");

				std::cout << fixed << " to " << moving << std::endl;

				typename FieldType::Pointer directDisplacementField { }, inverseDisplacementField { }, velocityField { };

				///////////////////////////////

				int numberOfIterations { m_FirstRegistration ? 10 : 1 };

				//int numberOfIterations { 1 };
				//int numberOfIterations {3};
				int numberOfLevels { 1 };
				int numberOfExponentiatorIterations { 4 };
				double timestep { 1.0 };

				bool useImageSpacing { true };

				// Regularizer parameters
				int regularizerType { 1 };       // Diffusive
				float regulAlpha = 0.5;
				float regulVar = 0.5;
				float regulMu = 0.5;
				float regulLambda = 0.5;

				int nccRadius { 2 };

				// Force parameters
				int forceType { 0 };              // Demon
				int forceDomain { 0 };            // Warped moving

				// Stop criterion parameters
				int stopCriterionPolicy { 1 }; // Simple graduated is default
				float stopCriterionSlope = 0.005;

				unsigned int its[numberOfLevels] { };
				its[numberOfLevels - 1] = numberOfIterations;
				for (int level = numberOfLevels - 2; level >= 0; --level) {
					its[level] = its[level + 1];
				}

				typedef itk::VariationalSymmetricDiffeomorphicRegistrationFilter<InputImageType, InputImageType, FieldType> VariationalFilterType;
				//typedef itk::VariationalDiffeomorphicRegistrationFilter<InputImageType, InputImageType, FieldType> VariationalFilterType;

				typename VariationalFilterType::Pointer regFilter = VariationalFilterType::New();

				regFilter->SetNumberOfExponentiatorIterations(numberOfExponentiatorIterations);

				typedef itk::VariationalRegistrationDiffusionRegularizer<FieldType> DiffusionRegularizerType;
				//typedef itk::VariationalRegistrationElasticRegularizer<FieldType> DiffusionRegularizerType;
				//typedef itk::VariationalRegistrationGaussianRegularizer<FieldType> DiffusionRegularizerType;

				typename DiffusionRegularizerType::Pointer regularizer = DiffusionRegularizerType::New();

				//regularizer->SetAlpha(0);
				//regularizer->SetAlpha(regularizer->GetAlpha() * pow(2,6));
				//regularizer->SetAlpha()
				//regularizer->SetStandardDeviations(25);
				//regularizer->SetMaximumKernelWidth(75);
				//regularizer->SetUseImageSpacing(false);
				regFilter->SetRegularizer(regularizer);

				//if(level==0 && it==0){
				//typedef itk::VariationalRegistrationNCCFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				//typedef ttt::VariationalRegistrationLogNCCFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				typedef itk::VariationalRegistrationDemonsFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				//typedef itk::VariationalRegistrationSSDFunction<InputImageType, InputImageType, FieldType> NCCFunctionType;
				typename NCCFunctionType::Pointer function = NCCFunctionType::New();
#if 1
				typename NCCFunctionType::RadiusType radius;
				radius.Fill(4);
				function->SetRadius(radius);
#endif
				regFilter->SetDifferenceFunction(function);

				typedef itk::VariationalRegistrationMultiResolutionFilter<InputImageType, InputImageType, FieldType> MRRegistrationFilterType;

				MRRegistrationFilterType::Pointer mrRegFilter = MRRegistrationFilterType::New();
				mrRegFilter->SetRegistrationFilter(regFilter);
				mrRegFilter->SetMovingImage(movingFrame);
				mrRegFilter->SetFixedImage(fixedFrame);

				mrRegFilter->SetNumberOfLevels(numberOfLevels);
				mrRegFilter->SetNumberOfIterations(its);

				if (!m_FirstRegistration) {

					typename FieldType::Pointer initialField { };
					initialField = m_Project.GetMotionField(fixed, moving, m_CurrentLevel);

					mrRegFilter->SetInitialField(initialField);
				}

				//
				// Setup stop criterion
				//
				typedef VariationalRegistrationStopCriterion<VariationalFilterType, MRRegistrationFilterType> StopCriterionType;
				StopCriterionType::Pointer stopCriterion = StopCriterionType::New();
				stopCriterion->SetRegressionLineSlopeThreshold(stopCriterionSlope);
				stopCriterion->PerformLineFittingMaxDistanceCheckOn();

				switch (stopCriterionPolicy) {
				case 1:
					stopCriterion->SetMultiResolutionPolicyToSimpleGraduated();
					break;
				case 2:
					stopCriterion->SetMultiResolutionPolicyToGraduated();
					break;
				default:
					stopCriterion->SetMultiResolutionPolicyToDefault();
					break;
				}

				regFilter->AddObserver(itk::IterationEvent(), stopCriterion);
				mrRegFilter->AddObserver(itk::IterationEvent(), stopCriterion);
				mrRegFilter->AddObserver(itk::InitializeEvent(), stopCriterion);
				//
				// Setup logger
				//
				typedef VariationalRegistrationLogger<VariationalFilterType, MRRegistrationFilterType> LoggerType;
				LoggerType::Pointer logger = LoggerType::New();

				regFilter->AddObserver(itk::IterationEvent(), logger);
				mrRegFilter->AddObserver(itk::IterationEvent(), logger);

				mrRegFilter->Update();

				directDisplacementField = mrRegFilter->GetDisplacementField();
				directDisplacementField->DisconnectPipeline();
				m_Project.SetMotionField(fixed, moving, m_CurrentLevel, directDisplacementField);

				inverseDisplacementField = regFilter->GetInverseDisplacementField();
				inverseDisplacementField->DisconnectPipeline();

				m_Project.SetMotionField(moving, fixed, m_CurrentLevel, inverseDisplacementField);

			} //Do registration
		}
		m_FirstRegistration = false;
	}

	void InitDeconvolutionIteration() {

		for (int t = m_First; t <= m_Last; t++) {

			//std::stringstream frameNum("");
			//			frameNum << "_T" << t;

			typename InputImageType::Pointer frame;

			frame = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename InputImageType::Pointer paddedFrame { }, paddedPSF { }, psf { };
			typename ComplexImageType::Pointer transformedFrame { };
			typename ComplexImageType::Pointer transferFunction { };
			psf = m_Project.GetPSF(t, m_CurrentLevel);

			PrepareImageAndPSF<InputImageType, ComplexImageType>(frame, psf, paddedFrame, transformedFrame, paddedPSF, transferFunction);

			m_Project.SetEstimatedFrequencyImage(t, m_CurrentLevel, transformedFrame);
			m_Project.SetTransferImage(t, m_CurrentLevel, transferFunction);
			//WriteFile<InputImageType>(std::string(directory),std::string("originalPSF"),frameNum.str(),"mha",paddedPSF);

			typename InputImageType::Pointer poissonLagrange = InputImageType::New();
			poissonLagrange->CopyInformation(frame);
			poissonLagrange->SetRegions(frame->GetLargestPossibleRegion());

			poissonLagrange->Allocate();
			poissonLagrange->FillBuffer(0.0);

			m_Project.SetPoissonLagrange(t, m_CurrentLevel, poissonLagrange);
			m_Project.SetBoundsLagrange(t, m_CurrentLevel, poissonLagrange);

			typename HessianImageType::Pointer hessianLagrange = HessianImageType::New();

			hessianLagrange->CopyInformation(frame);
			hessianLagrange->SetRegions(frame->GetLargestPossibleRegion());

			hessianLagrange->Allocate();
			hessianLagrange->FillBuffer(itk::NumericTraits<HessianType>::ZeroValue());

			m_Project.SetHessianLagrange(t, m_CurrentLevel, hessianLagrange);

			for (int t1 = t - m_W; t1 <= t + m_W; t1++) {
				m_Project.SetMovingLagrange(t1, t, m_CurrentLevel, poissonLagrange);
			}
		} //DO INIT
	}

	void DoReconstructStep() {
		for (int t = m_First; t <= m_Last; t++) {
			std::cout << "RECONSTRUCT T " << t << std::endl;
			typename InputImageType::Pointer conjugatedPoisson, paddedConjugatedPoisson, conjugatedHessian, paddedConjugatedHessian,
					conjugatedBounded, paddedConjugatedBounded, totalPadded, total, normalizer;

			typename ComplexImageType::Pointer conjugatedPoissonFrequency, conjugatedHessianFrequency, conjugatedBoundedFrequency, transfer,
					totalFrequency;

			typedef itk::AddImageFilter<ComplexImageType, ComplexImageType, ComplexImageType> ComplexAccumulator;
			typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> Accumulator;

			typename ComplexAccumulator::Pointer complexAccumulator = ComplexAccumulator::New();
			typename Accumulator::Pointer normalizerAccumulator = Accumulator::New();

			//UPDATES
			conjugatedPoisson = m_Project.GetConjugatedPoisson(t, m_CurrentLevel);
			PadImage<InputImageType>(conjugatedPoisson, m_PadSize, m_PadLowerBound, paddedConjugatedPoisson);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedPoisson, conjugatedPoissonFrequency);

			conjugatedHessianFrequency = m_Project.GetConjugatedHessian(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(conjugatedHessianFrequency, m_Offset, conjugatedHessianFrequency);
			conjugatedHessianFrequency->SetOrigin(conjugatedPoissonFrequency->GetOrigin());
#if 0
			PadImage<InputImageType>(conjugatedHessian, padSize, padLowerBound, paddedConjugatedHessian);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedHessian, conjugatedHessianFrequency);
#endif
			conjugatedBounded = m_Project.GetConjugatedBounded(t, m_CurrentLevel);
			PadImage<InputImageType>(conjugatedBounded, m_PadSize, m_PadLowerBound, paddedConjugatedBounded);
			ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedBounded, conjugatedBoundedFrequency);

			complexAccumulator->SetInput1(conjugatedPoissonFrequency);
			complexAccumulator->SetInput2(conjugatedHessianFrequency);
			complexAccumulator->Update();

			totalFrequency = complexAccumulator->GetOutput();
			totalFrequency->DisconnectPipeline();

			complexAccumulator->SetInput1(totalFrequency);
			complexAccumulator->SetInput2(conjugatedBoundedFrequency);
			complexAccumulator->Update();

			totalFrequency = complexAccumulator->GetOutput();
			totalFrequency->DisconnectPipeline();

			//NORMALIZERS

			//POISSON
			std::cout << "Poisson" << std::endl;
			transfer = m_Project.GetTransferImage(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);

			typedef itk::ComplexToModulusImageAdaptor<ComplexImageType, double> ModulusFilter;
			typename ModulusFilter::Pointer modulusAdaptor = ModulusFilter::New();
			modulusAdaptor->SetImage(transfer);

			typedef itk::SquareImageFilter<ModulusFilter, InputImageType> SquareFilter;
			typename SquareFilter::Pointer squareFilter = SquareFilter::New();
			squareFilter->SetInput(modulusAdaptor);
			squareFilter->Update();

			normalizer = squareFilter->GetOutput();
			normalizer->DisconnectPipeline();

			//HESSIAN
			std::cout << "Hessian" << std::endl;
			normalizerAccumulator->SetInput1(normalizer);
			m_HessianNormalizer->SetOrigin(normalizer->GetOrigin());
			normalizerAccumulator->SetInput2(m_HessianNormalizer);
			normalizerAccumulator->Update();

			normalizer = normalizerAccumulator->GetOutput();
			normalizer->DisconnectPipeline();

			//BOUNDS
			std::cout << "Bounds" << std::endl;
			typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
			typename AddType::Pointer addScalarFilter = AddType::New();
			addScalarFilter->SetInput1(normalizer);
			addScalarFilter->SetConstant2(1);
			addScalarFilter->Update();

			normalizer = addScalarFilter->GetOutput();
			normalizer->DisconnectPipeline();

			for (int t1 = t - m_W; t1 <= t + m_W; t1++) {

				if (t1 < m_First || t1 > m_Last || t1 == t)
					continue;

				typename InputImageType::Pointer conjugatedMovingFrame, paddedConjugatedMovingFrame;
				typename ComplexImageType::Pointer conjugatedMovingFrameFrequency;

				conjugatedMovingFrame = m_Project.GetMovingConjugated(t1, t, m_CurrentLevel);

				PadImage<InputImageType>(conjugatedMovingFrame, m_PadSize, m_PadLowerBound, paddedConjugatedMovingFrame);
				ImageFFT<InputImageType, ComplexImageType>(paddedConjugatedMovingFrame, conjugatedMovingFrameFrequency);

				complexAccumulator->SetInput1(totalFrequency);
				complexAccumulator->SetInput2(conjugatedMovingFrameFrequency);
				complexAccumulator->Update();

				totalFrequency = complexAccumulator->GetOutput();
				totalFrequency->DisconnectPipeline();

				normalizerAccumulator->SetInput1(normalizer);
				normalizerAccumulator->SetInput2(squareFilter->GetOutput());
				normalizerAccumulator->Update();

				normalizer = normalizerAccumulator->GetOutput();
				normalizer->DisconnectPipeline();

				typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
				typename AddType::Pointer addScalarFilter = AddType::New();
				addScalarFilter->SetInput1(normalizer);
				addScalarFilter->SetConstant2(1);
				addScalarFilter->Update();

				normalizer = addScalarFilter->GetOutput();
				normalizer->DisconnectPipeline();

			}

			typedef itk::DivideImageFilter<ComplexImageType, InputImageType, ComplexImageType> ComplexDividerType;

			typename ComplexDividerType::Pointer complexDivider = ComplexDividerType::New();
			totalFrequency->SetOrigin(normalizer->GetOrigin());
			complexDivider->SetInput1(totalFrequency);
			complexDivider->SetInput2(normalizer);
			complexDivider->Update();

			totalFrequency = complexDivider->GetOutput();
			totalFrequency->DisconnectPipeline();

			m_Project.SetEstimatedFrequencyImage(t, m_CurrentLevel, totalFrequency);

			ImageIFFT<InputImageType, ComplexImageType>(totalFrequency, totalPadded);
			CropImage<InputImageType>(totalPadded, conjugatedPoisson->GetLargestPossibleRegion(), total);
			total->GetOrigin().Fill(0);
			total->SetSpacing(conjugatedPoisson->GetSpacing());
			m_Project.SetEstimatedImage(t, m_CurrentLevel, total);

		}

	}

	void DoShrinkStep() {
		auto original = m_Project.GetOriginalImage(m_First, m_CurrentLevel);
		for (int t = m_First; t <= m_Last; t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;
			std::cout << "SHRINK T " << t << std::endl;

			typename InputImageType::Pointer original;
			original = m_Project.GetOriginalImage(t, m_CurrentLevel);

			typename InputImageType::Pointer estimate;
			estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename ComplexImageType::Pointer transfer;
			transfer = m_Project.GetTransferImage(t, m_CurrentLevel);

			AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);
			typename ComplexImageType::Pointer estimateFrequency { };

			typename InputImageType::Pointer paddedEstimate { };

			//typename InputImageType::SizeType padSize = GetPadSize<InputImageType>(original, psf);
			//typename InputImageType::SizeType padLowerBound = GetPadLowerBound<InputImageType>(original, psf);

			//estimateFrequency = project.GetEstimatedFrequencyImage(t,level);
			PadImage<InputImageType>(estimate, m_PadSize, m_PadLowerBound, paddedEstimate);

			ImageFFT<InputImageType, ComplexImageType>(paddedEstimate, estimateFrequency);

			typename InputImageType::Pointer poissonLagrange;

			poissonLagrange = m_Project.GetPoissonLagrange(t, m_CurrentLevel);
			//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);

			typename InputImageType::Pointer denoised { };
			typename InputImageType::Pointer conjugatedDenoised { };
			typename ComplexImageType::Pointer conjugatedDenoisedFrequency { };

			original->SetSpacing(m_OneSpacing);
			poissonLagrange->SetSpacing(m_OneSpacing);
			transfer->SetSpacing(m_OneSpacing);
			estimateFrequency->SetSpacing(m_OneSpacing);

			original->SetOrigin(m_ZeroOrigin);
			poissonLagrange->SetOrigin(m_ZeroOrigin);
			transfer->SetOrigin(m_ZeroOrigin);
			estimateFrequency->SetOrigin(m_ZeroOrigin);

			DoPoissonStage<InputImageType, ComplexImageType>(original, estimateFrequency, poissonLagrange, transfer, m_PadSize, m_PadLowerBound, denoised,
					conjugatedDenoisedFrequency);
			denoised->SetSpacing(m_OriginalSpacing);
			denoised->SetOrigin(m_OriginalOrigin);
			m_Project.SetPoissonShrinkedImage(t, m_CurrentLevel, denoised);
			//WriteFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);

			//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedDenoisedFrequency"),frameNum.str(),"mha",conjugatedDenoisedFrequency);

			ImageIFFT<InputImageType, ComplexImageType>(conjugatedDenoisedFrequency, conjugatedDenoised);
			CropImage<InputImageType>(conjugatedDenoised, original->GetLargestPossibleRegion(), conjugatedDenoised);
			conjugatedDenoised->SetSpacing(m_OriginalSpacing);
			conjugatedDenoised->SetOrigin(m_OriginalOrigin);
			m_Project.SetConjugatedPoisson(t, m_CurrentLevel, conjugatedDenoised);
			//WriteFile<InputImageType>(std::string(directory),std::string("conjugatedDenoised"),frameNum.str(),"mha",conjugatedDenoised);
		}

		for (int t = m_First; t <= m_Last; t++) {

			typename HessianImageType::Pointer hessianLagrange;

			hessianLagrange = m_Project.GetHessianLagrange(t, m_CurrentLevel);
			typename InputImageType::PointType originalOrigin = hessianLagrange->GetOrigin();

			typename InputImageType::SpacingType originalSpacing = hessianLagrange->GetSpacing();

			//ReadFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);

			typename InputImageType::PointType zeroOrigin;
			zeroOrigin.Fill(0.0);
			typename InputImageType::SpacingType oneSpacing;
			oneSpacing.Fill(1.0);

			hessianLagrange->SetOrigin(zeroOrigin);
			hessianLagrange->SetSpacing(oneSpacing);

			typename HessianImageType::Pointer shrinkedHessian { }, hessian { };
			typename ComplexImageType::Pointer conjugatedHessian { };
			typename ComplexImageType::Pointer estimateFrequency { };

			typename InputImageType::Pointer estimate;
			estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);

			typename InputImageType::Pointer paddedEstimate { };

			PadImage<InputImageType>(estimate, m_PadSize, m_PadLowerBound, paddedEstimate);

			ImageFFT<InputImageType, ComplexImageType>(paddedEstimate, estimateFrequency);


			DoHessianStage<ComplexImageType, HessianImageType, HessianComplexImageType>(estimateFrequency, hessianLagrange, m_HessianFilter,
					original->GetLargestPossibleRegion(), m_PadSize, m_PadLowerBound, hessian, shrinkedHessian, conjugatedHessian);

			InputImageType::Pointer conjugatedHessianAmplitude;

			ImageIFFT<InputImageType, ComplexImageType>(conjugatedHessian, conjugatedHessianAmplitude);
			CropImage<InputImageType>(conjugatedHessianAmplitude, original->GetLargestPossibleRegion(), conjugatedHessianAmplitude);

			m_Project.SetShrinkedHessian(t, m_CurrentLevel, shrinkedHessian);
			//WriteFile<HessianImageType>(std::string(directory),std::string("shrinkedHessian"),frameNum.str(),"mha",shrinkedHessian);

			conjugatedHessianAmplitude->SetSpacing(originalSpacing);
			conjugatedHessianAmplitude->SetOrigin(originalOrigin);

			m_Project.SetConjugatedHessian(t, m_CurrentLevel, conjugatedHessian);
		}

		for (int t = m_First; t <= m_Last; t++) {

			//		typename InputImageType::Pointer currentEstimate = ReadFile<InputImageType>(std::string(directory),std::string("estimate"),frameNum.str(),"mha");

			typename InputImageType::Pointer boundsLagrange { };
			boundsLagrange = m_Project.GetBoundsLagrange(t, m_CurrentLevel);

			//ReadFile<InputImageType>(std::string(directory),std::string("boundsLagrange"),frameNum.str(),"mha",boundsLagrange);

			typename InputImageType::Pointer bounded { };
			typename InputImageType::Pointer conjugatedBounded { };

			typename InputImageType::Pointer estimate = m_Project.GetEstimatedImage(t, m_CurrentLevel);
			DoBoundsStage<InputImageType, ComplexImageType>(estimate, boundsLagrange, m_PadSize, m_PadLowerBound, bounded, conjugatedBounded);

			m_Project.SetShrinkedBounded(t, m_CurrentLevel, bounded);
			//WriteFile<InputImageType>(std::string(directory),std::string("bounded"),frameNum.str(),"mha",bounded);

			m_Project.SetConjugatedBounded(t, m_CurrentLevel, conjugatedBounded);

			//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedBounded"),frameNum.str(),"mha",conjugatedBounded);

			for (int t1 = t - m_W; t1 <= t + m_W; t1++) {
				if (t1 < m_First || t1 > m_Last || t1 == t)
					continue;
				typedef itk::ContinuousBorderWarpImageFilter<InputImageType, InputImageType, FieldType> MovingImageWarperType;

				typename InputImageType::Pointer movingEstimate, shrinkedMoving, conjugatedMoving, lagrangeMoving;

				typename MovingImageWarperType::Pointer warper = MovingImageWarperType::New();

				typename FieldType::Pointer registrationField;
				registrationField = m_Project.GetMotionField(t1, t, m_CurrentLevel);

				warper->SetInput(estimate);
				warper->SetOutputParametersFromImage(estimate);
				warper->SetDisplacementField(registrationField);
				warper->Update();

				shrinkedMoving = warper->GetOutput();
				shrinkedMoving->DisconnectPipeline();

				m_Project.SetMovingShrinked(t, t1, m_CurrentLevel, shrinkedMoving);

				typedef itk::SubtractImageFilter<InputImageType, InputImageType, InputImageType> SubtractImageFilterType;

				typename SubtractImageFilterType::Pointer subtractor = SubtractImageFilterType::New();
				lagrangeMoving = m_Project.GetMovingLagrange(t, t1, m_CurrentLevel);
				lagrangeMoving->SetOrigin(shrinkedMoving->GetOrigin());
				subtractor->SetInput1(shrinkedMoving);
				subtractor->SetInput2(lagrangeMoving);

				subtractor->Update();

				conjugatedMoving = subtractor->GetOutput();
				m_Project.SetMovingConjugated(t, t1, m_CurrentLevel, conjugatedMoving);
			}

		}						//DO SHRINK

	}

	void DoInit(){
		auto original = m_Project.GetOriginalImage(m_First, m_CurrentLevel);

		typedef ttt::CentralDifferenceHessianSource<FloatType, 3> HessianSourceType;

		typename HessianSourceType::Pointer hessianSource = HessianSourceType::New();
		//hessianSource->SetNumberOfThreads(this->GetNumberOfThreads());
		hessianSource->SetLowerPad(m_PadLowerBound);
		hessianSource->SetPadSize(m_PadSize);

		typename InputImageType::SpacingType hessianSpacing;

		hessianSpacing[0] = 1;
		hessianSpacing[1] = 1;
		hessianSpacing[2] = original->GetSpacing()[2] / original->GetSpacing()[0];

		hessianSource->SetSpacing(hessianSpacing);
		hessianSource->Update();

		typename HessianComplexImageType::Pointer hessianFilter = hessianSource->GetOutput();
		hessianFilter->DisconnectPipeline();

		typedef ttt::TensorToEnergyImageFilter<HessianComplexImageType, InputImageType> HessianEnergyFilter;
		typename HessianEnergyFilter::Pointer hessianEnergyFilter1 = HessianEnergyFilter::New();
		hessianEnergyFilter1->SetInput(hessianFilter);
		hessianEnergyFilter1->Update();
		typename InputImageType::Pointer hessianNormalizer = hessianEnergyFilter1->GetOutput();
		hessianNormalizer->DisconnectPipeline();

	}
	void DoLagrangeStep(){


		typedef itk::ContinuousBorderWarpImageFilter<InputImageType, InputImageType, FieldType> MovingImageWarperType;


		for (int t = m_First; t <= m_Last; t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;

			typename ComplexImageType::Pointer resultFrequency, transfer;
			typename InputImageType::Pointer result, poissonLagrange, denoised;
			//std::stringstream frameNum("");
			//frameNum << "_T"<< t;

			result = m_Project.GetEstimatedImage(t, m_CurrentLevel);
			//ReadFile<InputImageType>(std::string(directory),std::string("estimate"),frameNum.str(),"mha",result);

			resultFrequency = m_Project.GetEstimatedFrequencyImage(t, m_CurrentLevel);
			//ReadFile<ComplexImageType>(std::string(directory),std::string("estimateFrequency"),frameNum.str(),"mha",resultFrequency);
			{
				transfer = m_Project.GetTransferImage(t, m_CurrentLevel);
				//ReadFile<ComplexImageType>(std::string(directory),std::string("transfer"),frameNum.str(),"mha",transfer);

				poissonLagrange = m_Project.GetPoissonLagrange(t, m_CurrentLevel);
				//ReadFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);
				denoised = m_Project.GetPoissonShrinkedImage(t, m_CurrentLevel);
				//ReadFile<InputImageType>(std::string(directory),std::string("denoised"),frameNum.str(),"mha",denoised);

				AdjustImage<ComplexImageType>(resultFrequency, m_Offset, resultFrequency);
				AdjustImage<ComplexImageType>(transfer, m_Offset, transfer);
				resultFrequency->GetOrigin().Fill(0);
				transfer->GetOrigin().Fill(0);
				denoised->GetOrigin().Fill(0);

				resultFrequency->SetSpacing(result->GetSpacing());
				transfer->SetSpacing(result->GetSpacing());

				typedef itk::MultiplyImageFilter<ComplexImageType, ComplexImageType, ComplexImageType> ComplexMultiplyType;

				ComplexMultiplyType::Pointer complexMultiplyFilter4 = ComplexMultiplyType::New();
				complexMultiplyFilter4->SetInput1(resultFrequency);
				complexMultiplyFilter4->SetInput2(transfer);

				typedef itk::HalfHermitianToRealInverseFFTImageFilter<ComplexImageType, InputImageType> IFFTFilterType;
				typename IFFTFilterType::Pointer IFFTFilter3 = IFFTFilterType::New();
				IFFTFilter3->SetInput(complexMultiplyFilter4->GetOutput());
				IFFTFilter3->Update();
				InputImageType::Pointer tmp = IFFTFilter3->GetOutput();

				CropImage<InputImageType>(tmp, result->GetLargestPossibleRegion(), tmp);
				tmp->GetOrigin().Fill(0);
				poissonLagrange->GetOrigin().Fill(0);
				tmp->SetSpacing(result->GetSpacing());
				poissonLagrange->SetSpacing(result->GetSpacing());

				typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> AddType;
				typename AddType::Pointer addFilter3 = AddType::New();
				addFilter3->SetInput1(tmp);
				addFilter3->SetInput2(poissonLagrange);

				typedef itk::SubtractImageFilter<InputImageType, InputImageType, InputImageType> SubType;
				typename SubType::Pointer subFilter3 = SubType::New();
				subFilter3->SetInput1(addFilter3->GetOutput());
				subFilter3->SetInput2(denoised);
				subFilter3->Update();
				poissonLagrange = subFilter3->GetOutput();
				poissonLagrange->DisconnectPipeline();

				m_Project.SetPoissonLagrange(t, m_CurrentLevel, poissonLagrange);
				//WriteFile<InputImageType>(std::string(directory),std::string("poissonLagrange"),frameNum.str(),"mha",poissonLagrange);

			}

			//////////////////
			{
				typename HessianImageType::Pointer hessianLagrange, shrinkedHessian;
				hessianLagrange = m_Project.GetHessianLagrange(t, m_CurrentLevel);
				hessianLagrange->GetOrigin().Fill(0);
				//ReadFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);
				shrinkedHessian = m_Project.GetShrinkedHessian(t, m_CurrentLevel);
				//ReadFile<HessianImageType>(std::string(directory),std::string("shrinkedHessian"),frameNum.str(),"mha",shrinkedHessian);

				typedef ttt::MultiplyByTensorImageFilter<ComplexImageType, HessianComplexImageType> HessianFrequencyFilterType;

				typedef ttt::HessianIFFTImageFilter<HessianComplexImageType, HessianImageType> HessianIFFTFilterType;
				typedef itk::ExtractImageFilter<HessianImageType, HessianImageType> ExtractFilterType;
				typedef ttt::HessianFFTImageFilter<HessianImageType, HessianComplexImageType> HessianFFTFilterType;

				typedef itk::AddImageFilter<HessianImageType> AddHessianType;
				typedef itk::SubtractImageFilter<HessianImageType> SubHessianType;

				typename HessianFrequencyFilterType::Pointer hessianFrequency = HessianFrequencyFilterType::New();
				resultFrequency->GetSpacing().Fill(1);
				hessianFrequency->SetInput1(resultFrequency);
				hessianFrequency->SetInput2(m_HessianFilter);

				typename HessianIFFTFilterType::Pointer hessianIFFT = HessianIFFTFilterType::New();
				hessianIFFT->SetInput(hessianFrequency->GetOutput());
				hessianIFFT->Update();
				HessianImageType::Pointer hessian = hessianIFFT->GetOutput();
				CropImage<HessianImageType>(hessian, result->GetLargestPossibleRegion(), hessian);

				hessian->SetSpacing(result->GetSpacing());
				hessian->SetOrigin(result->GetOrigin());
				//WriteFile<ComplexImageType>(std::string(directory),std::string("conjugatedHessian"),frameNum.str(),"mha",conjugatedHessian);
				//WriteFile<HessianImageType>(std::string(directory),std::string("hessian"),frameNum.str(),"mha",hessian);
				m_Project.SetHessian(t, m_CurrentLevel, hessian);

#if 1
				typedef itk::UnaryFunctorImageFilter<HessianImageType, InputImageType, PlatenessFunctor<typename HessianImageType::PixelType> > PlatenessImageFilterType;

				typename PlatenessImageFilterType::Pointer platenessFilter = PlatenessImageFilterType::New();

				platenessFilter->SetInput(hessian);
				platenessFilter->Update();
				typename InputImageType::Pointer plateness = platenessFilter->GetOutput();
				plateness->DisconnectPipeline();

				WriteFile<InputImageType>(std::string(m_Directory), std::string("plateness"), frameNum.str(), "mha", plateness);
#endif

				typename AddHessianType::Pointer adder = AddHessianType::New();
				adder->SetInput1(hessianLagrange);

				adder->SetInput2(hessian);

				typename SubHessianType::Pointer subtractor = SubHessianType::New();
				subtractor->SetInput1(adder->GetOutput());
				shrinkedHessian->SetSpacing(result->GetSpacing());
				subtractor->SetInput2(shrinkedHessian);
				subtractor->Update();
				hessianLagrange = subtractor->GetOutput();
				hessianLagrange->DisconnectPipeline();

				m_Project.SetHessianLagrange(t, m_CurrentLevel, hessianLagrange);
				//WriteFile<HessianImageType>(std::string(directory),std::string("hessianLagrange"),frameNum.str(),"mha",hessianLagrange);
			}

			typedef itk::AddImageFilter<InputImageType, InputImageType> AddType;
			typedef itk::SubtractImageFilter<InputImageType, InputImageType> SubType;

			typename InputImageType::Pointer boundsLagrange { };
			boundsLagrange = m_Project.GetBoundsLagrange(t, m_CurrentLevel);
			boundsLagrange->GetOrigin().Fill(0);
			//ReadFile<InputImageType>(std::string(directory),std::string("boundsLagrange"),frameNum.str(),"mha",boundsLagrange);

			typename InputImageType::Pointer bounded { };
			bounded = m_Project.GetShrinkedBounded(t, m_CurrentLevel);
			bounded->GetOrigin().Fill(0);
			//ReadFile<InputImageType>(std::string(directory),std::string("bounded"),frameNum.str(),"mha",bounded);

			typename AddType::Pointer adder = AddType::New();
			adder->SetInput1(result);
			adder->SetInput2(boundsLagrange);
			typename SubType::Pointer subtractor = SubType::New();

			subtractor->SetInput1(adder->GetOutput());
			subtractor->SetInput2(bounded);
			subtractor->Update();
			boundsLagrange = subtractor->GetOutput();
			boundsLagrange->DisconnectPipeline();
			m_Project.SetBoundsLagrange(t, m_CurrentLevel, boundsLagrange);

			for (int t1 = t - m_W; t1 <= t + m_W; t1++) {

				if (t1 < m_First || t1 > m_Last || t1 == t)
					continue;

				typename InputImageType::Pointer shrinkedMoving, lagrangeMoving;

				shrinkedMoving = m_Project.GetMovingShrinked(t1, t, m_CurrentLevel);
				lagrangeMoving = m_Project.GetMovingLagrange(t1, t, m_CurrentLevel);

				shrinkedMoving->GetOrigin().Fill(0);
				lagrangeMoving->GetOrigin().Fill(0);

				typename AddType::Pointer adder = AddType::New();
				adder->SetInput1(result);
				adder->SetInput2(lagrangeMoving);

				typename SubType::Pointer subtractor = SubType::New();

				subtractor->SetInput1(adder->GetOutput());
				subtractor->SetInput2(shrinkedMoving);
				subtractor->Update();
				lagrangeMoving = subtractor->GetOutput();
				lagrangeMoving->DisconnectPipeline();

				m_Project.SetMovingLagrange(t1, t, m_CurrentLevel, lagrangeMoving);

			}

		}
	}

	void DoBlind(){
		for (int t = m_First; t <= m_Last; t++) {
			std::stringstream frameNum("");
			frameNum << "_T" << t;
			typename ComplexImageType::Pointer imageEstimateFrequency, transferCurrentEstimateFrequency, transferNextEstimateFrequency;
			typename InputImageType::Pointer original, paddedOriginal, estimatedPSF, image, paddedImage;
			original = m_Project.GetOriginalImage(t, m_CurrentLevel);
			image = m_Project.GetEstimatedImage(t, m_CurrentLevel);
			//imageEstimateFrequency = project.GetEstimatedFrequencyImage(t, level);
			//AdjustImage<ComplexImageType>(imageEstimateFrequency, offset, imageEstimateFrequency);

			transferCurrentEstimateFrequency = m_Project.GetTransferImage(t, m_CurrentLevel);
			AdjustImage<ComplexImageType>(transferCurrentEstimateFrequency, m_Offset, transferCurrentEstimateFrequency);

			typename InputImageType::SpacingType oneSpacing = original->GetSpacing();
			typename InputImageType::PointType zeroOrigin = original->GetOrigin();
			//typename InputImageType::SpacingType originalSpacing = original->GetSpacing();

			oneSpacing.Fill(1);
			zeroOrigin.Fill(0);

			original->SetSpacing(oneSpacing);

			transferCurrentEstimateFrequency->SetSpacing(oneSpacing);
			//imageEstimateFrequency->SetSpacing(oneSpacing);

			transferCurrentEstimateFrequency->SetOrigin(zeroOrigin);
			//imageEstimateFrequency->SetOrigin(zeroOrigin);
			original->SetOrigin(zeroOrigin);

			PadImage<InputImageType>(image, m_PadSize, m_PadLowerBound, paddedImage);

			typedef itk::MaximumImageFilter<InputImageType> MaximumType;

			typename MaximumType::Pointer maximum = MaximumType::New();
			maximum->SetInput1(paddedImage);
			maximum->SetConstant2(0.0);
			maximum->Update();
			ImageFFT<InputImageType, ComplexImageType>(maximum->GetOutput(), imageEstimateFrequency);

			PadImage<InputImageType>(original, m_PadSize, m_PadLowerBound, paddedOriginal);

			BlindRL<InputImageType, ComplexImageType>(transferCurrentEstimateFrequency, imageEstimateFrequency, paddedOriginal, estimatedPSF,
					transferNextEstimateFrequency);
			WriteFile<InputImageType>(std::string(m_Directory), std::string("estimatedPSF"), frameNum.str(), "mha", estimatedPSF);
			m_Project.SetTransferImage(t, m_CurrentLevel, transferNextEstimateFrequency);
			//BlindRL(const typename TComplexImage::Pointer & currentTransferEstimate, const typename TComplexImage::Pointer & currentImageEstimate, const typename TImage::Pointer & paddedOriginal, typename TImage::Pointer & transferNext){
		} //DO BLIND
	}

	void DoUpsample() {
		//UPSAMPLE
		int nextLevel = m_CurrentLevel + 1;
		typename InputImageType::SpacingType nextSpacing = m_Project.GetOriginalImage(m_First, nextLevel)->GetSpacing();
		typename InputImageType::SizeType nextSize = m_Project.GetOriginalImage(m_First, nextLevel)->GetLargestPossibleRegion().GetSize();

		for (int t = m_First; t <= m_Last; t++) {

			typedef itk::ResampleImageFilter<InputImageType, InputImageType> ResampleFilterType;
			ResampleFilterType::Pointer resampler = ResampleFilterType::New();
			typedef itk::LinearInterpolateImageFunction<InputImageType, double> Interpolator;
			typedef itk::NearestNeighborExtrapolateImageFunction<InputImageType, double> Extrapolator;
			typename Interpolator::Pointer interpolator = Interpolator::New();
			typename Extrapolator::Pointer extrapolator = Extrapolator::New();

			resampler->SetInterpolator(interpolator);
			resampler->SetExtrapolator(extrapolator);
			resampler->SetInput(m_Project.GetEstimatedImage(t, m_CurrentLevel));
			resampler->SetSize(nextSize);
			resampler->SetOutputSpacing(nextSpacing);
			resampler->Update();
			m_Project.SetEstimatedImage(t, nextLevel, resampler->GetOutput());

			typedef itk::VectorResampleImageFilter<FieldType, FieldType> FieldExpanderType;

			typename FieldExpanderType::Pointer fieldExpander = FieldExpanderType::New();
			fieldExpander->SetSize(nextSize);
			fieldExpander->SetOutputSpacing(nextSpacing);

			for (int t1 = t - m_W; t1 <= t + m_W; t1++) {
				if (t1 < m_First || t1 > m_Last || t1 == t)
					continue;

				typename FieldType::Pointer registrationField, nextField;
				registrationField = m_Project.GetMotionField(t, t1, m_CurrentLevel);

				fieldExpander->SetInput(registrationField);
				fieldExpander->UpdateLargestPossibleRegion();

				nextField = fieldExpander->GetOutput();
				nextField->DisconnectPipeline();
				m_Project.SetMotionField(t1, t, nextLevel, nextField);
			}

		} //UPSAMPLE
	}

	void DeconvIter(){
		this->DoShrinkStep();
		this->DoReconstructStep();
		this->DoLagrangeStep();
	}

	void DeconvLoop(){

		this->InitDeconvolutionIteration();
		for(int it=0;it<m_MaxDeconvolutionIterations;it++){
			this->DeconvIter();
		}
	}
public:
	void MainLoop(){
		this->InitPyramid();
		for(int level=0;level<m_NumberOfLevels;level++){
			for(int it=0;it< m_NumberOfIterations;it++){
				this->DoRegistration();
				this->DeconvLoop();
				this->DoBlind();
			}
			this->DoUpsample();
		}
	}
};

#define DO_INIT
#define DO_REGISTRATION
#define DO_DECONVOLUTION
#define DO_SHRINK

#define DO_RECONSTRUCT
#define DO_BLIND
#define DO_LAGRANGE


int main(int argc, char ** argv) {

	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <directory> <prefix>  <first> <last>" << std::endl;
	}

	unsigned int MaxIterations = 3;
	unsigned int MaxOuterIterations = 20;
	char * directory = argv[1];
	char * prefix = argv[2];
	int first = atoi(argv[3]);
	int last = atoi(argv[4]);

	TrackingAndDeconvolutionProject project { };
	project.NewProject(0, (last - first), std::string(directory));

	typedef double FloatType;
	typedef itk::Image<FloatType, 3> InputImageType;
	typedef itk::Image<std::complex<FloatType>, 3> ComplexImageType;
	typedef itk::Image<itk::Vector<FloatType, 3>, 3> FieldType;

	typedef itk::SymmetricSecondRankTensor<FloatType, 3> HessianType;
	typedef itk::Image<HessianType, 3> HessianImageType;

	typedef itk::Image<itk::SymmetricSecondRankTensor<std::complex<FloatType>, 3>, 3> HessianComplexImageType;
	int W { 1 };

	//InputImageType::SpacingType spacing;
	//spacing.Fill(1);
	//psf->SetSpacing(spacing);
	unsigned int numberOfLevels = 6;


	//2. Main loop
	//For each level

	bool firstRegistration = true;

	HSPVRRAL2 deconvoluter(project);
	deconvoluter.MainLoop();

	for (int level = 0; level < numberOfLevels; level++) {

		for (int outerit = 0; outerit < MaxOuterIterations; outerit++) {
			std::cout << "OuterIt: " << outerit << std::endl;

			//2.1.2 Deconvolve image

			//firstRegistration = true;

			typename InputImageType::Pointer original;
			std::stringstream frameNum("");
			frameNum << "_T" << first;
			original = project.GetOriginalImage(first, level);

			typename InputImageType::SpacingType oneSpacing = original->GetSpacing();
			typename InputImageType::SpacingType originalSpacing = original->GetSpacing();

			oneSpacing.Fill(1);

			typename InputImageType::PointType zeroOrigin = original->GetOrigin();
			typename InputImageType::PointType originalOrigin = original->GetOrigin();
			zeroOrigin.Fill(0);

			typename InputImageType::Pointer psf = project.GetPSF(first, level);

			typename InputImageType::SizeType padSize = GetPadSize<InputImageType>(original, psf);
			typename InputImageType::SizeType padLowerBound = GetPadLowerBound<InputImageType>(original, psf);

			typename InputImageType::OffsetType offset;
			offset[0] = -padLowerBound[0];
			offset[1] = -padLowerBound[1];
			offset[2] = -padLowerBound[2];

		} //OuterIt
		  //2.2 Upsample



	} //level

} //MAIN

#endif
