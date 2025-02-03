#include <SFML/Graphics.hpp>
#include <immintrin.h>  // AVX2 fejléckönyvtár
#include <iostream>

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 1000;

// Kezdeti tartomány
double minRe = -2.0, maxRe = 1.0;
double minIm = -1.5, maxIm = 1.5;

// Színkódolás az iterációk alapján
sf::Color getColor(int iteration) {
    if (iteration == MAX_ITER) return sf::Color::Black;
    return sf::Color(iteration % 256, (iteration * 2) % 256, (iteration * 5) % 256);
}

// Mandelbrot számítás AVX2-vel
void generateMandelbrot(sf::Image &image) {
    double reFactor = (maxRe - minRe) / WIDTH;
    double imFactor = (maxIm - minIm) / HEIGHT;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x += 4) {  // 4 pixel egyszerre
            __m256d c_re = _mm256_set_pd(
                minRe + (x + 3) * reFactor,
                minRe + (x + 2) * reFactor,
                minRe + (x + 1) * reFactor,
                minRe + x * reFactor
            );
            __m256d c_im = _mm256_set1_pd(minIm + y * imFactor);
            __m256d z_re = _mm256_set1_pd(0.0);
            __m256d z_im = _mm256_set1_pd(0.0);
            __m256d four = _mm256_set1_pd(4.0);
            __m256d two = _mm256_set1_pd(2.0);
            __m256d iter = _mm256_set1_pd(0);
            __m256d maxIter = _mm256_set1_pd(MAX_ITER);

            for (int i = 0; i < MAX_ITER; i++) {
                __m256d z_re2 = _mm256_mul_pd(z_re, z_re);
                __m256d z_im2 = _mm256_mul_pd(z_im, z_im);
                __m256d abs_val = _mm256_add_pd(z_re2, z_im2);

                __m256d mask = _mm256_cmp_pd(abs_val, four, _CMP_LT_OQ);
                if (_mm256_movemask_pd(mask) == 0) break;

                __m256d z_re_new = _mm256_add_pd(_mm256_sub_pd(z_re2, z_im2), c_re);
                __m256d z_im_new = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(z_re, z_im), two), c_im);

                z_re = _mm256_blendv_pd(z_re, z_re_new, mask);
                z_im = _mm256_blendv_pd(z_im, z_im_new, mask);
                iter = _mm256_add_pd(iter, _mm256_and_pd(mask, _mm256_set1_pd(1.0)));
            }

            double results[4];
            _mm256_storeu_pd(results, iter);

            for (int i = 0; i < 4; i++) {
                if (x + i < WIDTH) {
                    image.setPixel(x + i, y, getColor(static_cast<int>(results[i])));
                }
            }
        }
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot AVX2 + Zoom");
    sf::Image image;
    image.create(WIDTH, HEIGHT, sf::Color::Black);

    generateMandelbrot(image);

    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    bool redraw = false;
    double zoomFactor = 0.7;
    double moveStep = 0.1;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            // Zoom egérgörgővel
            if (event.type == sf::Event::MouseWheelScrolled) {
                double zoom = (event.mouseWheelScroll.delta > 0) ? zoomFactor : (1.0 / zoomFactor);
                double mouseRe = minRe + (maxRe - minRe) * event.mouseWheelScroll.x / WIDTH;
                double mouseIm = minIm + (maxIm - minIm) * event.mouseWheelScroll.y / HEIGHT;

                minRe = mouseRe + (minRe - mouseRe) * zoom;
                maxRe = mouseRe + (maxRe - mouseRe) * zoom;
                minIm = mouseIm + (minIm - mouseIm) * zoom;
                maxIm = mouseIm + (maxIm - mouseIm) * zoom;

                redraw = true;
            }

            // Mozgatás nyílbillentyűkkel
            if (event.type == sf::Event::KeyPressed) {
                double reOffset = (maxRe - minRe) * moveStep;
                double imOffset = (maxIm - minIm) * moveStep;

                if (event.key.code == sf::Keyboard::Left) {
                    minRe -= reOffset;
                    maxRe -= reOffset;
                }
                if (event.key.code == sf::Keyboard::Right) {
                    minRe += reOffset;
                    maxRe += reOffset;
                }
                if (event.key.code == sf::Keyboard::Up) {
                    minIm -= imOffset;
                    maxIm -= imOffset;
                }
                if (event.key.code == sf::Keyboard::Down) {
                    minIm += imOffset;
                    maxIm += imOffset;
                }

                redraw = true;
            }
        }

        // Ha zoom vagy mozgás történt, újra kell számolni
        if (redraw) {
            generateMandelbrot(image);
            texture.loadFromImage(image);
            redraw = false;
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}

