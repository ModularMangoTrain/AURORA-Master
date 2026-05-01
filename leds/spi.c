#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <stdio.h>

#define NUM_LEDS 144

void ws2812_byte_to_spi(uint8_t byte, uint8_t *spi_buf) {
    int i;
    for (i = 0; i < 8; i++) {
        spi_buf[i] = (byte & 0x80) ? 0b110 : 0b100;
        byte <<= 1;
    }
}

int main(void) {
    int spi_fd = open("/dev/spidev1.0", O_RDWR);
    if (spi_fd < 0) {
        printf("Failed to open SPI1\n");
        return 1;
    }

    uint32_t speeds[] = {800000, 1000000, 1200000, 1600000, 2000000, 2400000};
    int speed_count = 6;
    
    for (int s = 0; s < speed_count; s++) {
        printf("Testing speed: %d Hz\n", speeds[s]);
        
        uint8_t mode = SPI_MODE_0;
        ioctl(spi_fd, SPI_IOC_WR_MODE, &mode);
        ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speeds[s]);

        uint8_t led_data[24];
        uint8_t buffer[NUM_LEDS * 24];
        
        ws2812_byte_to_spi(0, &led_data[0]);
        ws2812_byte_to_spi(255, &led_data[8]);
        ws2812_byte_to_spi(0, &led_data[16]);
        
        for (int i = 0; i < NUM_LEDS; i++) {
            memcpy(&buffer[i * 24], led_data, 24);
        }
        
        write(spi_fd, buffer, sizeof(buffer));
        sleep(3);
    }

    close(spi_fd);
    return 0;
}
