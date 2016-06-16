/*
 * splitsource.cpp
 *
 *  Created on: Jan 20, 2015
 *      Author: morgan
 */

#include <itkSCIFIOImageIO.h>
#include <itkVectorImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkExtractImageFilter.h>
#include <sstream>
#include <iomanip>
#include "itkVectorIndexSelectionCastImageFilter.h"
int main(int argc,char ** argv){
	std::string inputFile(argv[1]);
	std::string outputPrefix(argv[2]);

	typedef itk::SCIFIOImageIO InputImageIOType;

	typedef itk::Image<double,4> InputImageType;
	typedef itk::Image<double,3> OutputImageType;

	typedef itk::ImageFileReader<InputImageType> InputImageReaderType;

	typename InputImageReaderType::Pointer inputReader = InputImageReaderType::New();

	typename InputImageIOType::Pointer inputImageIO = InputImageIOType::New();

	inputReader->SetImageIO(inputImageIO);
	inputReader->SetFileName(inputFile);
	inputReader->UpdateOutputInformation();

	typename InputImageType::Pointer input=inputReader->GetOutput();

	unsigned int numFrames = input->GetLargestPossibleRegion().GetSize(3);

	typedef itk::ExtractImageFilter<InputImageType,OutputImageType> ExtractImageFilterType;
	typename ExtractImageFilterType::Pointer extractor= ExtractImageFilterType::New();
	extractor->SetInput(inputReader->GetOutput());

	InputImageType::IndexType extractionIndex=inputReader->GetOutput()->GetLargestPossibleRegion().GetIndex();
	InputImageType::SizeType extractionSize=inputReader->GetOutput()->GetLargestPossibleRegion().GetSize();
	extractionSize[3]=0;

	typedef itk::ImageFileWriter<OutputImageType> WriterType;
	WriterType::Pointer writer = WriterType::New();


	writer->SetInput(extractor->GetOutput());

	for(int t=0;t<numFrames;t++){
		extractionIndex[3]=t;
		typename InputImageType::RegionType extractionRegion(extractionIndex,extractionSize);
		extractor->SetExtractionRegion(extractionRegion);

		extractor->SetDirectionCollapseToIdentity();
		std::stringstream buffer("");
		buffer << outputPrefix << "_T" << t << ".mha";
		writer->SetFileName(buffer.str());
		writer->Update();
	}


#if 0
	std::string inputPrefix(argv[1]);
	std::string inputSufix(argv[2]);
	std::string outputPrefix(argv[3]);
	std::string outputSufix(argv[4]);
	int firstFrame = atoi(argv[5]);
	int lastFrame = atoi(argv[6]);


	typedef itk::SCIFIOImageIO InputImageIOType;
	typedef itk::MetaImageIO OutputImageIO;

	typedef itk::VectorImage<double,3> InputImageType;
	typedef itk::Image<double,3> OutputImageType;

	typedef itk::ImageFileReader<InputImageType> InputImageReaderType;

	typename InputImageReaderType::Pointer inputReader = InputImageReaderType::New();

	InputImageIOType::Pointer inputIO= InputImageIOType::New();



	for(int t=firstFrame;t<=lastFrame;t++){
		std::stringstream formatInput("");
		formatInput << inputPrefix << std::setfill ('0') << std::setw (3) << t << inputSufix;
		inputReader->SetImageIO(inputIO);
		inputReader->SetFileName(formatInput.str());

		//inputReader->Update();


		//std::cout << input << std::endl;

		typedef itk::ImageFileWriter<OutputImageType> OutputImageWriterType;
		typename OutputImageWriterType::Pointer writerA= OutputImageWriterType::New();
		typename OutputImageWriterType::Pointer writerB= OutputImageWriterType::New();

		typename OutputImageIO::Pointer outputImageIO= OutputImageIO::New();

		std::stringstream formatOutputA("");
		formatOutputA << outputPrefix << "actin_T" << t << outputSufix;

		std::stringstream formatOutputB("");
		formatOutputB << outputPrefix << "myoII_T" << t << outputSufix;

		typedef itk::VectorIndexSelectionCastImageFilter<InputImageType,OutputImageType> SelectionType;

		typename SelectionType::Pointer selectorA = SelectionType::New();
		typename SelectionType::Pointer selectorB = SelectionType::New();

		selectorA->SetInput(inputReader->GetOutput());
		selectorA->SetIndex(1);



		writerA->SetFileName(formatOutputA.str());
		writerA->SetInput(selectorA->GetOutput());
		writerA->Update();
		std::cout << inputReader->GetOutput()->GetNumberOfComponentsPerPixel() << std::endl;

		selectorB->SetInput(inputReader->GetOutput());
		selectorB->SetIndex(2);

		writerB->SetFileName(formatOutputB.str());
		writerB->SetInput(selectorB->GetOutput());
		writerB->Update();
	}




#if 0
	typedef itk::ImageFileWriter<OutputImageType> OutputImageWriterType;
	typename OutputImageWriterType::Pointer writer= OutputImageWriterType::New();

	typename OutputImageIO::Pointer outputImageIO= OutputImageIO::New();

	writer->SetImageIO(outputImageIO);
	writer->SetFileName(buffer.str());
	writer->SetInput(extractor->GetOutput());
	writer->Update();
#endif

	#if 1

#endif
#endif
}
