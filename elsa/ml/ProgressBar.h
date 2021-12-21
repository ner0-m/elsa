// (The MIT License)
//
// Copyright (c) 2016 Prakhar Srivastav <prakhar@prakhar.me>
// Copyright (c) 2019 David Tellenbach <david.tellenbach@in.tum.de>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// 'Software'), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/**
 * @file ProgressBar.h
 * @brief Implements a progessbar
 * @author Prakhar Srivastav <prakhar@prakhar.me>
 * @author David Tellenbach <david.tellenbach@in.tum.de>
 */
#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>

namespace elsa::ml
{
    namespace detail
    {
        /**
         * \ingroup Internal
         * \class ProgressBar
         * @brief A progress bar
         *
         * Usage example:
         *
         * ```
         * uint32_t total = 100;
         * ProgressBar bar(total, 50);
         * for (uint32_t i = 0; i < 100; ++t) {
         *   // Do something
         *   ++bar;
         *   bar.display();
         * }
         * bar.done();
         * ```
         *
         * Output:
         *
         * ```
         * [========================>                         ] 50% 0.034s
         * [==================================================] 100% 0.094s
         * ```
         */
        class ProgressBar
        {
        public:
            /**
             * @brief Constructor
             * @param total The total number of ticks in the progress bar
             * @param width The width of the progress bar in chars
             * @param complete The char that will be displayed to indicate already
             * completed parts of the progress bar. This is optional and defaults to '='
             * @param incomplete The char that will be displayed to indicate yet
             * uncompleted parts of the progress bar. This is optional and defaults to ' '
             * @param head The char that will be displayed the head of the progress bar.
             * This is optional and defaults to '>'
             */
            ProgressBar(uint32_t total, uint32_t width, char complete = '=', char incomplete = ' ',
                        char head = '>')
                : total_ticks(total),
                  bar_width(width),
                  complete_char(complete),
                  incomplete_char(incomplete),
                  head_char(head)
            {
            }

            /**
             * @brief Increment the progress bar
             * @return The value of \p ticks after incrementing it
             */
            uint32_t operator++() { return ++ticks; }

            /**
             * @brief Display the progress bar
             * @param preMessage A message to display right before the progressbar
             * @param postMessage A message to display right after the progressbar
             */
            void display(const std::string& preMessage = std::string(""),
                         const std::string& postMessage = std::string("")) const
            {
                float progress = static_cast<float>(ticks) / static_cast<float>(total_ticks);
                uint32_t pos = static_cast<uint32_t>(static_cast<float>(bar_width) * progress);

                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                auto time_elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

                std::cout << preMessage << "[";

                for (uint32_t i = 0; i < bar_width; ++i) {
                    if (i < pos)
                        std::cout << complete_char;
                    else if (i == pos)
                        std::cout << head_char;
                    else
                        std::cout << incomplete_char;
                }
                std::cout << "] " << static_cast<int>(progress * 100.f) << "% "
                          << std::setprecision(2) << std::fixed
                          << static_cast<float>(time_elapsed) / 1000.f << "s ";
                std::cout << postMessage << "\r";
                std::cout.flush();
            }

            /**
             * @brief Indicate that the progressbar has finished
             * @param preMessage A message to display right before the progressbar
             * @param postMessage A message to display right after the progressbar
             */
            void done(const std::string& preMessage = std::string(""),
                      const std::string& postMessage = std::string(""))
            {
                ticks = total_ticks;
                display(preMessage, postMessage);
                ticks = 0;
                start_time = std::chrono::steady_clock::now();
                std::cout << std::endl;
            }

        private:
            uint32_t ticks = 0;
            const uint32_t total_ticks;
            const uint32_t bar_width;
            const char complete_char;
            const char incomplete_char;
            const char head_char;
            std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        };
    } // namespace detail
} // namespace elsa::ml
