#include "./TimeTracker.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <string>

using namespace std;

TimeTracker::TimeTracker(const string& name) {
    name_ = name;
    start();
}

TimeTracker::~TimeTracker() {

}

void TimeTracker::start() {
    start_time_ = chrono::steady_clock::now();
    cout << "############# Started Time Tracking: " << name_ << " #############" << endl;
}

long TimeTracker::stop() {
    end_time_ = chrono::steady_clock::now();

    long duration = chrono::duration_cast<chrono::milliseconds>(end_time_ - start_time_).count();

    cout << "Duration: " << duration << " ms" << endl;
    cout << "############# Stopped Time Tracking: " << name_ << " #############" << endl;
    return duration;
}