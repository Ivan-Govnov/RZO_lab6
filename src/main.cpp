#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main()
{
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;
    cv::CascadeClassifier smileCascade;

    if (!faceCascade.load("haarcascade_frontalface_default.xml"))
    {
        std::cout << "Cannot load face cascade\n";
        return -1;
    }

    if (!eyeCascade.load("haarcascade_eye.xml"))
    {
        std::cout << "Cannot load eye cascade\n";
        return -1;
    }

    if (!smileCascade.load("haarcascade_smile.xml"))
    {
        std::cout << "Cannot load smile cascade\n";
        return -1;
    }

    cv::VideoCapture cap("ZUA.mp4");

    if (!cap.isOpened())
    {
        std::cout << "Cannot open video\n";
        return -1;
    }

    cv::Mat frame;
    cv::Mat gray;
    bool save_image = false;
    int count = 0;

    double fps = 0.0;

while (cap.read(frame)) {
    double _start = (double)cv::getTickCount();
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(50, 50));


    for (auto face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        cv::Mat faceROI = gray(face);

        // глаза
        std::vector<cv::Rect> eyes, eyesFiltered;
        eyeCascade.detectMultiScale(faceROI, eyes, 1.2, 8, 0,
                                    cv::Size(20, 20), cv::Size(face.width/3, face.height/3));

        for (auto eye : eyes) {
            int eyeCenterY = eye.y + eye.height/2;
            if (eyeCenterY < face.height / 2) {
                eyesFiltered.push_back(eye);
            }
        }
        if (eyesFiltered.size() > 2) {
            std::sort(eyesFiltered.begin(), eyesFiltered.end(),
                      [](const cv::Rect& a, const cv::Rect& b) { return a.area() > b.area(); });
            eyesFiltered.resize(2);
        }


        for (auto eye : eyesFiltered) {
            cv::Point center(face.x + eye.x + eye.width/2,
                             face.y + eye.y + eye.height/2);
            int radius = (eye.width + eye.height)/4;
            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
        }

        // улыбка
        std::vector<cv::Rect> smiles, smilesFiltered;
        smileCascade.detectMultiScale(faceROI, smiles, 1.3, 20, 0,
                                      cv::Size(25, 20), cv::Size(face.width/2, face.height/3));

        for (auto smile : smiles) {
            int smileCenterY = smile.y + smile.height/2;
            if (smileCenterY > face.height / 2) {
                smilesFiltered.push_back(smile);
            }
        }
        if (smilesFiltered.size() > 1) {
            std::sort(smilesFiltered.begin(), smilesFiltered.end(),
                      [](const cv::Rect& a, const cv::Rect& b) { return a.area() > b.area(); });
            smilesFiltered.resize(1);
        }

        for (auto smile : smilesFiltered) {
            cv::Rect smileRect(face.x + smile.x, face.y + smile.y,
                               smile.width, smile.height);
            cv::rectangle(frame, smileRect, cv::Scalar(0, 255, 255), 2);
        }
    }

    double _end = (double)cv::getTickCount();
    double fps = cv::getTickFrequency() / (_end - _start);
    cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Face / Eyes / Smile detection", frame);
    count += 1;
    if (count == 1 || count == 10 || count == 100) {
            save_image = true;
    }
    if (save_image == true) {
        std::string name = "result" + std::to_string(count) + ".jpg";
        cv::imwrite(name, frame);
        save_image = false;
    }
    if (cv::waitKey(30) == 27) break;
}

    cap.release();
    cv::destroyAllWindows();

    return 0;
}