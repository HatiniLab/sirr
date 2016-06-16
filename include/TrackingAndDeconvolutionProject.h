/*
 * TrackingAndDeconvolutionProject.h
 *
 *  Created on: Feb 11, 2015
 *      Author: morgan
 */
#include <itkImage.h>
#include <itkSymmetricSecondRankTensor.h>
//#include <vtk_jsoncpp.h>


class TrackingAndDeconvolutionProject{

public:
	static const unsigned int dim =3;
	typedef double FloatType;
	typedef itk::Image<FloatType,dim> ImageType;
	typedef itk::Image<std::complex<FloatType>,dim> ComplexImageType;
	typedef itk::Image<itk::Vector<FloatType,dim>, dim > FieldImageType;
	typedef itk::Image<itk::SymmetricSecondRankTensor<FloatType,dim>, dim > HessianImageType;
	typedef itk::Image<itk::SymmetricSecondRankTensor<std::complex<FloatType>,dim>, dim > HessianComplexImageType;


	typename ImageType::Pointer GetOriginalImage(int frame);

	void SetObservedImage(int frame,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetObservedImage(int frame,int level);


	typename ImageType::Pointer GetTemplatePSF();


	void SetPSF(int frame,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetPSF(int frame,int level);


	void SetEstimatedImage(int frame,int level, const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetEstimatedImage(int frame,int level);

	void SetEstimatedFrequencyImage( int frame,int level,const typename ComplexImageType::Pointer & original);
	typename ComplexImageType::Pointer GetEstimatedFrequencyImage(int frame,int level);

	void SetTransferImage(int frame,int level,const typename ComplexImageType::Pointer & transfer);
	typename ComplexImageType::Pointer GetTransferImage(int frame,int level );


	void SetPoissonShrinkedImage(int frame,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetPoissonShrinkedImage(int frame,int level);

	void SetWarpedPoissonShrinkedImage(int frame0,int frame1,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetWarpedPoissonShrinkedImage(int frame0, int frame1,int level);

	void SetConjugatedPoisson(int frame,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetConjugatedPoisson(int frame,int level);

	void SetConjugatedWarpedPoisson(int frame0,int frame1,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetConjugatedWarpedPoisson(int frame0,int frame1,int level);


	void SetMovingShrinked(int frame0,int frame1,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetMovingShrinked(int frame0,int frame1,int level);

	void SetMovingConjugated(int frame0,int frame1,int level,const typename ImageType::Pointer & original);
	typename ImageType::Pointer GetMovingConjugated(int frame0,int frame1,int level);


	void SetHessian(int frame, int level,const typename HessianImageType::Pointer & hessian);
	typename HessianImageType::Pointer GetHessian(int frame,int level);

	void SetShrinkedHessian(int frame,int level,const typename HessianImageType::Pointer & hessianImage);
	typename HessianImageType::Pointer GetShrinkedHessian( int frame,int level);

	void SetConjugatedHessian(int frame,int level,const typename ComplexImageType::Pointer & conjugated);
	typename ComplexImageType::Pointer GetConjugatedHessian(int frame,int level);

	void SetShrinkedBounded(int frame,int level,const typename ImageType::Pointer & bounded);
	typename ImageType::Pointer GetShrinkedBounded(int frame,int level);

	void SetConjugatedBounded(int frame,int level,const typename ImageType::Pointer & conjugated);
	typename ImageType::Pointer GetConjugatedBounded(int frame,int level);

	void SetMotionField(int frame0, int frame1,int level,const typename FieldImageType::Pointer & motionField);
	typename FieldImageType::Pointer GetMotionField(int frame0,int frame1,int level);


	void SetPoissonLagrange(int frame,int level,const typename ImageType::Pointer & lagrange );
	typename ImageType::Pointer GetPoissonLagrange(int frame,int level);

	void SetWarpedPoissonLagrange(int frame0,int frame1,int level,const typename ImageType::Pointer & lagrange );
	typename ImageType::Pointer GetWarpedPoissonLagrange(int frame0,int frame1,int level);

	void SetHessianLagrange(int frame,int level, const typename HessianImageType::Pointer & hessian);
	typename HessianImageType::Pointer GetHessianLagrange(int frame,int level);

	void SetBoundsLagrange(int frame,int level,const typename ImageType::Pointer & bounds);
	typename ImageType::Pointer GetBoundsLagrange(int frame,int level);

	void SetMovingLagrange(int frame0,int frame1,int level,const typename ImageType::Pointer & movingLagrange);
	typename ImageType::Pointer GetMovingLagrange(int frame0,int frame1,int level);

	void NewProject(int firstFrame, int lastFrame,const std::string & path,const std::string & originalName);

	void OpenProject(const std::string & path);

	unsigned int GetNumberOfFrames(){
		return m_NumFrames;
	}

private:

    template<class T> void ReadFrame(typename T::Pointer & result,const std::string & name);
	template<class T> void ReadFrameSCIFIO(typename T::Pointer & result, const std::string & name);

    template<class T> void WriteFrame(const typename T::Pointer & image,const std::string & name);


	std::string m_ProjectPath;
	std::string m_OriginalName;
	int m_FirstFrame;
	int m_LastFrame;
	unsigned int m_NumFrames;


};
