#include <iostream>
#include <fstream>
#include <array>
#include <typeinfo>
#include <math.h>

#include "Eigen.h"

#include "VirtualSensor.h"

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

double edgeLength(Vector4f pos1, Vector4f pos2) 
{
	double d[] = {abs(pos1(0) - pos2(0)), abs(pos1(1) - pos2(1)), abs(pos1(2) - pos2(2))};
	double d_square[] = {d[0]*d[0], d[1]*d[1], d[2]*d[2]};
	double distance = sqrt(d_square[0] + d_square[1] + d_square[2]);
	return distance;
}

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// Get number of vertices
	unsigned int nVertices = width * height;
	
	// TODO: Determine number of valid faces
	unsigned nFaces = 0;
	int faces [(width-1) * (height-1) * 2][3] = {};
	int count = 0;
	for (int i=0; i < height-1; i++) {
		for (int j=0; j < width-1; j++) {
			int idx = i*width + j;
			int rightIdx = i*width + (j+1);
			int diagIdx = (i+1)*width + (j+1);
			int botIdx = (i+1)*width + j;
			
			Vector4f pos = vertices[idx].position;
			Vector4f rightPos = vertices[rightIdx].position;
			Vector4f diagPos = vertices[diagIdx].position;
			Vector4f botPos = vertices[botIdx].position;
								
			if ( (pos != Vector4f(MINF, MINF, MINF, MINF)) && (rightPos != Vector4f(MINF, MINF, MINF, MINF)) &&
			     (botPos != Vector4f(MINF, MINF, MINF, MINF)) ) {
					 // Compute edge length
					 double cur2rightEdge = edgeLength(pos, rightPos);
					 double right2botEdge = edgeLength(rightPos, botPos);
					 double bot2curEdge = edgeLength(botPos, pos);
					 // Check for edgeThreshold
					 if ( (cur2rightEdge <= edgeThreshold) && (right2botEdge <= edgeThreshold) &&
					      (bot2curEdge <= edgeThreshold) ) {
							  faces[nFaces][0] = idx;
							  faces[nFaces][1] = rightIdx;
							  faces[nFaces][2] = botIdx;
							  nFaces++;  
						  }
				 }
			if ( (rightPos != Vector4f(MINF, MINF, MINF, MINF)) && (diagPos != Vector4f(MINF, MINF, MINF, MINF)) &&
			          (botPos != Vector4f(MINF, MINF, MINF, MINF)) ) {
						// Compute edge length
					 	double right2diagEdge = edgeLength(rightPos, diagPos);
					 	double diag2botEdge = edgeLength(diagPos, botPos);
						double bot2rightEdge = edgeLength(botPos, rightPos);
						// Check for edgeThreshold
						if ( (right2diagEdge <= edgeThreshold) && (diag2botEdge <= edgeThreshold) &&
						     (bot2rightEdge <= edgeThreshold) ) {
								 faces[nFaces][0] = rightIdx;
								 faces[nFaces][1] = diagIdx;
								 faces[nFaces][2] = botIdx;
								 nFaces++;
							 }
					  }
			count++;
		}
	}
	std::cout << "Count: " << count << std::endl;

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	// outFile << "# numVertices numFaces numEdges" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// save vertices
	// outFile << "# list of vertices" << std::endl;
	// outFile << "# X Y Z R G B A" << std::endl;
	for (int i=0; i < nVertices; i++) {
		Vector4f pos = vertices[i].position;
		Vector4uc color = vertices[i].color;
		if ( pos == Vector4f(MINF, MINF, MINF, MINF) ) {
			pos = Vector4f(0, 0, 0, 0);
		}
		outFile << double(pos(0)) << " " << double(pos(1)) << " " << double(pos(2)) << " " 
		        << int(color(0)) << " " << int(color(1)) << " " << int(color(2)) << " " << int(color(3)) << std::endl;
	}

	// save valid faces
	// outFile << "# list of faces" << std::endl;
	// outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
	for (int j=0; j < nFaces; j++) {
		outFile << 3 << " " << int(faces[j][0]) << " " << int(faces[j][1]) << " " 
		        << int(faces[j][2]) << std::endl;
	}

	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		float fX = depthIntrinsics(0, 0);
		float fY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		// Back-Projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];

		int depthImageWidth = sensor.GetDepthImageWidth();
		int depthImageHeight = sensor.GetDepthImageHeight();
		// std::cout << "Depth img width: " << depthImageWidth
		// 		  << ", Depth img height: "<< depthImageHeight << std::endl;

		for( int u = 0; u < depthImageHeight; u++) {
			for(int v = 0; v < depthImageWidth; v++) {
				float pixelValue = *(depthMap + u*depthImageWidth + v);
				if(pixelValue == MINF) {
					vertices[u*depthImageWidth + v].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[u*depthImageWidth + v].color = Vector4uc(0,0,0,0);
				}
				else {
                    
					// from P_pixel -> P_image -> P_camera
					float zI = pixelValue;
                    float zC = zI
					float xC = (u - cX) * zI / fX;
					float yC = (v - cY) * zI / fY;
					
					// from P_camera to P_world
					Vector4f pCamera(xC, yC, zC, 1.0);
					Vector4f pWorld = trajectoryInv * depthExtrinsicsInv * pCamera;
					// std::cout << pWorld(0) << "," << pWorld(1) << "," << pWorld(2) << ", " << pWorld(3) << std::endl;

					// Store in vertices
					int idx = u*depthImageWidth + v;
					vertices[idx].position = pWorld;
					vertices[idx].color = Vector4uc(int(colorMap[4*idx]), int(colorMap[4*idx + 1]),
					                                int(colorMap[4*idx + 2]), int(colorMap[4*idx + 3]));
					// color pixels method compare with http://nicolas.burrus.name/index.php/Research/KinectCalibration
				}
			}

		}

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}
