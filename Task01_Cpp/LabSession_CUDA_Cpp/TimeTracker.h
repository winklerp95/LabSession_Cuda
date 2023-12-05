#pragma once
#include <string>
#include <chrono>
#include <fstream>

using namespace std;

class TimeTracker {
public:
    TimeTracker(const string& name);
    ~TimeTracker();

    void start();
    long stop();

private:
    string name_;
    chrono::steady_clock::time_point start_time_;
    chrono::steady_clock::time_point end_time_;
};