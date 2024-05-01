#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <mutex>

class Stack {
public:
    std::vector<std::pair<int, int>> item;

    void push(std::pair<int, int> value) {
        item.push_back(value);
    }

    std::pair<int, int> pop() {
        std::pair<int, int> value = item.back();
        item.pop_back();
        return value;
    }

    size_t size() const {
        return item.size();
    }

    bool isEmpty() const {
        return item.empty();
    }

    void clear() {
        item.clear();
    }
};

class RegionGrow {
private:
    cv::Mat im;
    int h, w;
    std::vector<std::vector<int>> passedBy;
    int currentRegion;
    int iterations;
    std::vector<std::vector<cv::Vec3b>> SEGS;
    Stack stack;
    float thresh;
    std::mt19937 rng;
    std::mutex mtx;
    int rows;
    int cols;

    void readImage(const std::string& img_path) {
        im = cv::imread(img_path, cv::IMREAD_COLOR);
        if (im.empty()) {
            std::cerr << "Error: Image at " << img_path << " could not be read." << std::endl;
            exit(1);
        }
    }

    std::vector<std::pair<int, int>> getNeighbour(int x0, int y0) const {
        static const int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
        static const int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

        std::vector<std::pair<int, int>> neighbours;
        for (int k = 0; k < 8; ++k) {
            int x = x0 + dx[k];
            int y = y0 + dy[k];
            if (boundaries(x, y)) {
                neighbours.emplace_back(x, y);
            }
        }
        return neighbours;
    }

    std::vector<std::pair<int, int>> createSeeds() {
        std::vector<std::pair<int, int>> seeds;

        int part_height = h / rows;
        int part_width = w / cols;
        std::uniform_int_distribution<> distribHeight(0, part_height - 1);
        std::uniform_int_distribution<> distribWidth(0, part_width - 1);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int start_row = i * part_height;
                int start_col = j * part_width;
                int seed_x = distribHeight(rng) + start_row;
                int seed_y = distribWidth(rng) + start_col;
                seeds.emplace_back(seed_x, seed_y);
            }
        }
        return seeds;
    }

    bool passedAll(int max_iteration = 200000) const {
        return iterations > max_iteration || std::all_of(passedBy.begin(), passedBy.end(), [](const std::vector<int>& row) {
            return std::all_of(row.begin(), row.end(), [](int val) { return val > 0; });
        });
    }

    bool boundaries(int x, int y) const {
        return 0 <= x && x < h && 0 <= y && y < w;
    }

    float meanPixelValue(int x, int y) const {
        cv::Vec3b pixel = im.at<cv::Vec3b>(x, y);
        return (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
    }

    float variance(const std::vector<float>& elems) const {
        float mean = std::accumulate(elems.begin(), elems.end(),0.0f) / elems.size();
            return std::accumulate(elems.begin(), elems.end(), 0.0f, [mean](float acc, float x) {
            return acc + std::pow(x - mean, 2);
            }) / elems.size();
        }
    float distance(int x, int y, int x0, int y0) const {
        cv::Vec3b pixel0 = im.at<cv::Vec3b>(x0, y0);
        cv::Vec3b pixel = im.at<cv::Vec3b>(x, y);
        return std::sqrt(std::pow(pixel0[0] - pixel[0], 2) + std::pow(pixel0[1] - pixel[1], 2) +
                        std::pow(pixel0[2] - pixel[2], 2));
    }

    void colorPixel(int i, int j) {
        int val = passedBy[i][j];
        SEGS[i][j] = (val == 0) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(val * 35 % 256, val * 90 % 256, val * 30 % 256);
    }

    void display() const {
        cv::Mat segsArray(h, w, CV_8UC3);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                segsArray.at<cv::Vec3b>(i, j) = SEGS[i][j];
            }
        }
        cv::imshow("Segmented Image", segsArray);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void bfs(int x0, int y0) {
        std::lock_guard<std::mutex> lock(mtx); // Lock for thread safety
        int regionNum = passedBy[x0][y0];
        std::vector<float> elems = {meanPixelValue(x0, y0)};

        float var = thresh;
        auto neighbours = getNeighbour(x0, y0);

        for (const auto& point : neighbours) {
            int x = point.first;
            int y = point.second;
            if (passedBy[x][y] == 0 && distance(x, y, x0, y0) < var) {
                if (passedAll()) {
                    break;
                }
                passedBy[x][y] = regionNum;
                stack.push(point);
                elems.push_back(meanPixelValue(x, y));
                var = variance(elems);
            }
            var = std::max(var, thresh);
        }
    }

    // New method to process a single region in a separate thread
    void processRegion(int x0, int y0) {
        if (passedBy[x0][y0] == 0) {
            currentRegion++;
            passedBy[x0][y0] = currentRegion;
            stack.push(std::make_pair(x0, y0));

            while (!stack.isEmpty()) {
                std::pair<int, int> point = stack.pop();
                bfs(point.first, point.second);
                iterations++;
            }
        }
    }
public:
    RegionGrow(const std::string& im_path, float th, int rows, int cols) : rng(std::random_device{}()) {
        readImage(im_path);
        h = im.rows;
        w = im.cols;
        this->rows = rows;
        this->cols = cols;
        passedBy.resize(h, std::vector<int>(w, 0));
        currentRegion = 0;
        iterations = 0;
        SEGS.resize(h, std::vector<cv::Vec3b>(w, cv::Vec3b(0, 0, 0)));
        thresh = th;
    }

    void applyRegionGrow(bool cv_display = true) {
        std::vector<std::pair<int, int>> randomSeeds = createSeeds();
        std::shuffle(randomSeeds.begin(), randomSeeds.end(), rng);

        // Limit the number of threads to the number of available cores
        unsigned int nThreads = std::thread::hardware_concurrency();
        nThreads = nThreads == 0 ? 2 : nThreads; // Fallback in case hardware_concurrency returns 0
        std::vector<std::thread> threads;
        threads.reserve(nThreads);

        // Assign seeds to threads in a round-robin fashion
        for (size_t i = 0; i < randomSeeds.size(); ++i) {
            if (threads.size() < nThreads) {
                // Create a new thread
                threads.emplace_back(&RegionGrow::processRegion, this, randomSeeds[i].first, randomSeeds[i].second);
            } else {
                // Join a finished thread and reuse it
                threads[i % nThreads].join();
                threads[i % nThreads] = std::thread(&RegionGrow::processRegion, this, randomSeeds[i].first, randomSeeds[i].second);
            }
        }

        // Ensure all threads are completed
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        if (cv_display) {
            // Consider parallelizing this loop for large images
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    colorPixel(i, j);
                }
            }
            display();
        }
    }
    };

int main() {
    RegionGrow example("C:/Users/ACER/Desktop/TP2image/regionGrowingThread/lena.png", 10, 3, 2);
    example.applyRegionGrow();
    return 0;
}