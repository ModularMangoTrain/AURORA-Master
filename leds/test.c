#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <signal.h>
#include <stdlib.h>

#define NUM_LEDS 30
#define SPI_DEVICE "/dev/spidev0.0"
#define SPI_SPEED 2400000

int spi_fd;
volatile sig_atomic_t running = 1;

void sigint_handler(int sig) {
    running = 0;
}

void ws2812_byte_to_spi(uint8_t byte, uint8_t *spi_buf) {
    int i;
    for (i = 0; i < 8; i++) {
        spi_buf[i] = (byte & 0x80) ? 0b110 : 0b100;
        byte <<= 1;
    }
}

int main(void) {
    signal(SIGINT, sigint_handler);

    spi_fd = open(SPI_DEVICE, O_RDWR);
    if (spi_fd < 0) return 1;

    uint8_t mode = SPI_MODE_0;
    uint32_t speed = SPI_SPEED;
    ioctl(spi_fd, SPI_IOC_WR_MODE, &mode);
    ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);

    uint8_t led_data[24];
    uint8_t buffer[NUM_LEDS * 24];
    uint16_t i;

    ws2812_byte_to_spi(0, &led_data[0]);
    ws2812_byte_to_spi(255, &led_data[8]);
    ws2812_byte_to_spi(0, &led_data[16]);

    for (i = 0; i < NUM_LEDS; i++) {
        memcpy(&buffer[i * 24], led_data, 24);
    }

    while (running) {
        write(spi_fd, buffer, sizeof(buffer));
        usleep(30000);
    }

    ws2812_byte_to_spi(191, &led_data[0]);
    ws2812_byte_to_spi(255, &led_data[8]);
    ws2812_byte_to_spi(0, &led_data[16]);

    for (i = 0; i < NUM_LEDS; i++) {
        memcpy(&buffer[i * 24], led_data, 24);
    }
    write(spi_fd, buffer, sizeof(buffer));
    usleep(50000);

    close(spi_fd);
    return 0;
}
