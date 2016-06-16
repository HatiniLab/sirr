#include "TrackingAndDeconvolutionProject.h"
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkSCIFIOImageIO.h>

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetOriginalImage(int frame){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << m_OriginalName<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetObservedImage(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
    std::stringstream buffer("");
    buffer << "observed_L" << level <<"_T" << m_FirstFrame+frame <<".mha";
    std::string fileName = buffer.str();
    this->WriteFrame<ImageType>(image,fileName);
}



typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetObservedImage(int frame,int level){

	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "observed_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetPSF(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
    std::stringstream buffer("");
    buffer << "psf_L" << level <<"_T" << m_FirstFrame+frame <<".mha";
    std::string fileName = buffer.str();
    this->WriteFrame<ImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetTemplatePSF(){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "psf.ome.tif";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}


typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetPSF(int frame,int level){

	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "psf_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}




void TrackingAndDeconvolutionProject::SetEstimatedImage(int frame,int level, const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
    std::stringstream buffer("");
    buffer << "estimate_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";
    std::string fileName = buffer.str();
    this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetEstimatedImage(int frame,int level){

	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "estimate_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;

}

void TrackingAndDeconvolutionProject::SetEstimatedFrequencyImage( int frame,int level,const typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "estimateFrequency_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ComplexImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer TrackingAndDeconvolutionProject::GetEstimatedFrequencyImage(int frame,int level){
	typename ComplexImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "estimateFrequency_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ComplexImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetTransferImage(int frame,int level,const typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "transfer_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ComplexImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer TrackingAndDeconvolutionProject::GetTransferImage(int frame,int level ){
	typename ComplexImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "transfer_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ComplexImageType>(result,fileName);
    return result;
}


void TrackingAndDeconvolutionProject::SetPoissonShrinkedImage(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "poissonShrinked_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetPoissonShrinkedImage(int frame,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "poissonShrinked_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetWarpedPoissonShrinkedImage(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "warpedPoissonShrinked_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";


    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetWarpedPoissonShrinkedImage(int frame0,int frame1,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image ){
	std::stringstream buffer("");
    buffer << "warpedPoissonShrinked_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}



void TrackingAndDeconvolutionProject::SetConjugatedPoisson(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "conjugatedPoisson_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetConjugatedPoisson(int frame,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "conjugatedPoisson_L"<< level<<"_T" << m_FirstFrame+frame <<".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetConjugatedWarpedPoisson(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "conjugatedWarpedPoisson_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";


    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetConjugatedWarpedPoisson(int frame0,int frame1,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image ){
	std::stringstream buffer("");
    buffer << "conjugatedWarpedPoisson_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}

void TrackingAndDeconvolutionProject::SetMovingConjugated(int frame0,int frame1,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "conjugatedMoving_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetMovingConjugated(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "conjugatedMoving_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetMovingShrinked(int frame0,int frame1,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
	buffer << "shrinkedMoving_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetMovingShrinked(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "shrinkedMoving_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}


void TrackingAndDeconvolutionProject::SetHessian(int frame,int level,const typename TrackingAndDeconvolutionProject::HessianImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "hessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<HessianImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::HessianImageType::Pointer TrackingAndDeconvolutionProject::GetHessian(int frame,int level){
	typename HessianImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "hessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<HessianImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetShrinkedHessian(int frame,int level,const typename TrackingAndDeconvolutionProject::HessianImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "shrinkedHessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<HessianImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::HessianImageType::Pointer TrackingAndDeconvolutionProject::GetShrinkedHessian( int frame,int level){
	typename HessianImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "shrinkedHessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<HessianImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetConjugatedHessian(int frame,int level,const typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "conjugatedHessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ComplexImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::ComplexImageType::Pointer TrackingAndDeconvolutionProject::GetConjugatedHessian(int frame,int level){
	typename ComplexImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "conjugatedHessian_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ComplexImageType>(result,fileName);
    return result;
}


void TrackingAndDeconvolutionProject::SetShrinkedBounded(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "shrinkedBounded_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetShrinkedBounded(int frame,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "shrinkedBounded_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetConjugatedBounded(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "conjugatedBounded_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetConjugatedBounded(int frame,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "conjugatedBounded_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetMotionField(int frame0, int frame1,int level,const typename TrackingAndDeconvolutionProject::FieldImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "registration_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<FieldImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::FieldImageType::Pointer TrackingAndDeconvolutionProject::GetMotionField(int frame0,int frame1,int level){
	typename FieldImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "registration_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<FieldImageType>(result,fileName);
    return result;
}


void TrackingAndDeconvolutionProject::SetPoissonLagrange(int frame,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image ){
	std::stringstream buffer("");
    buffer << "poissonLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetPoissonLagrange(int frame,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "poissonLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}

typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetWarpedPoissonLagrange(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "warpedPoissonLagrange_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";


    std::string fileName=buffer.str();
    this->ReadFrame<ImageType>(result,fileName);
    return result;
}
void TrackingAndDeconvolutionProject::SetWarpedPoissonLagrange(int frame0,int frame1,int level,const typename TrackingAndDeconvolutionProject::ImageType::Pointer & image ){
	std::stringstream buffer("");
    buffer << "warpedPoissonLagrange_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
void TrackingAndDeconvolutionProject::SetHessianLagrange(int frame,int level, const typename TrackingAndDeconvolutionProject::HessianImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "hessianLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<HessianImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::HessianImageType::Pointer TrackingAndDeconvolutionProject::GetHessianLagrange(int frame,int level){
	typename HessianImageType::Pointer result{};

    std::stringstream buffer("");
    buffer << "hessianLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

    std::string fileName=buffer.str();
    this->ReadFrame<HessianImageType>(result,fileName);
    return result;
}

void TrackingAndDeconvolutionProject::SetBoundsLagrange( int frame,int level,const typename ImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "boundsLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetBoundsLagrange(int frame,int level){
	typename ImageType::Pointer result{};

	std::stringstream buffer("");
	buffer << "boundsLagrange_L"<< level<<"_T" << m_FirstFrame+frame<< ".mha";

	std::string fileName=buffer.str();
	this->ReadFrame<ImageType>(result,fileName);
	return result;
}

void TrackingAndDeconvolutionProject::SetMovingLagrange( int frame0,int frame1,int level,const typename ImageType::Pointer & image){
	std::stringstream buffer("");
    buffer << "movingLagrange_L"<< level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";
	std::string fileName = buffer.str();
	this->WriteFrame<ImageType>(image,fileName);
}
typename TrackingAndDeconvolutionProject::ImageType::Pointer TrackingAndDeconvolutionProject::GetMovingLagrange(int frame0,int frame1,int level){
	typename ImageType::Pointer result{};

	std::stringstream buffer("");
	buffer << "movingLagrange_L"<<  level<<"_FT" << m_FirstFrame+frame0 <<"_MT" << m_FirstFrame+frame1 << ".mha";

	std::string fileName=buffer.str();
	this->ReadFrame<ImageType>(result,fileName);
	return result;
}

void TrackingAndDeconvolutionProject::NewProject(int firstFrame, int lastFrame,const std::string & path,const std::string & originalName){
	this->m_FirstFrame=firstFrame;
	this->m_LastFrame=lastFrame;
	this->m_ProjectPath=path;
	this->m_OriginalName=originalName;
	this->m_NumFrames=lastFrame-firstFrame+1;
}

void TrackingAndDeconvolutionProject::OpenProject(const std::string & path){
	this->m_ProjectPath=path;
}

template<class T> void TrackingAndDeconvolutionProject::ReadFrame(typename T::Pointer & result,const std::string & name){
    std::stringstream buffer("");
    buffer << this->m_ProjectPath << "/" <<name;
    typedef itk::ImageFileReader<T> ReaderType;

    typename ReaderType::Pointer reader = ReaderType::New();
//	typedef itk::SCIFIOImageIO ImageIOType;
//	typename ImageIOType::Pointer imageIO=ImageIOType::New();

//	reader->SetImageIO(imageIO);
    std::cout << buffer.str() << std::endl;
    reader->SetFileName(buffer.str());
    reader->Update();
    result = reader->GetOutput();
    assert(result);
    result->DisconnectPipeline();

}

template<class T> void TrackingAndDeconvolutionProject::ReadFrameSCIFIO(typename T::Pointer & result, const std::string & name){
	std::stringstream buffer("");
	buffer << this->m_ProjectPath << "/" << name;
	typedef itk::ImageFileReader<T> ReaderType;
	typename ReaderType::Pointer reader = ReaderType::New();
	typedef itk::SCIFIOImageIO ImageIOType;
	typename ImageIOType::Pointer imageIO = ImageIOType::New();
	reader->SetImageIO(imageIO);
	reader->SetFileName(buffer.str());
	reader->Update();
	result=reader->GetOutput();
	assert(result);
	result->DisconnectPipeline();
}

template<class T> void TrackingAndDeconvolutionProject::WriteFrame(const typename T::Pointer & image,const std::string & name){
    std::stringstream buffer("");
    buffer << this->m_ProjectPath << "/" <<name;
    typedef itk::ImageFileWriter<T> WriterType;

    typename WriterType::Pointer writer = WriterType::New();

    std::cout << buffer.str() << std::endl;
    writer->SetFileName(buffer.str());
    writer->SetInput(image);
    writer->Update();
}
