#include "stm32l0xx_hal.h"
#include "stm32l0xx_it.h"
#include "stdio.h"
#include "stdbool.h"
#include "string.h"

int _write(int file, char *ptr, int len)
{
    int DataIdx;
    for (DataIdx = 0; DataIdx < len; DataIdx++)
    {
        while ((USART4->ISR & 0X40) == 0)
            ;                          /* 等待上一个字符发送完成 */
        USART4->TDR = (uint8_t)*ptr++; /* 将要发送的字符 ch 写入到DR寄存器 */
    }
    return len;
}

void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
    GPIO_InitTypeDef GPIO_InitStruct;

    /*##-1-启用外围设备和GPIO时钟#################################*/
    /* 启用GPIO发送/接收时钟*/
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* 启用USARTx时钟*/
    __HAL_RCC_USART4_CLK_ENABLE();

    GPIO_InitStruct.Pin = GPIO_PIN_10 | GPIO_PIN_11;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF6_USART4;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    // HAL_NVIC_EnableIRQ(USART2_IRQn);         /* 使能USART1中断通道 */
    // HAL_NVIC_SetPriority(USART2_IRQn, 3, 3); /* 抢占优先级3，子优先级3 */
}

void UART_init()
{
    UART_HandleTypeDef UartHandle;
    UartHandle.Instance = USART4;
    UartHandle.Init.BaudRate = 115200;
    UartHandle.Init.WordLength = UART_WORDLENGTH_8B;
    UartHandle.Init.StopBits = UART_STOPBITS_1;
    UartHandle.Init.Parity = UART_PARITY_NONE;
    UartHandle.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    UartHandle.Init.Mode = UART_MODE_TX_RX;
    if (HAL_UART_DeInit(&UartHandle) != HAL_OK)
    {
        Error_Handler();
    }
    if (HAL_UART_Init(&UartHandle) != HAL_OK)
    {
        Error_Handler();
    }
}

void LED_GPIO_Init()
{
    __HAL_RCC_GPIOA_CLK_ENABLE(); // GPIO A 时钟初始化

    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct); // GPIO初始化
}

void KEY_GPIO_init()
{
    __HAL_RCC_GPIOA_CLK_ENABLE(); // GPIO A 时钟初始化
    __HAL_RCC_GPIOA_CLK_ENABLE(); // GPIO B 时钟初始化

    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct); // GPIO初始化

    GPIO_InitStruct.Pin = GPIO_PIN_3 | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_10;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct); // GPIO初始化
}

void LED_Task()
{
    static uint32_t Time = 0;
    if (HAL_GetTick() - Time >= 2000)
    {
        Time = HAL_GetTick();
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5); /* LED0 状态取反 */
        // HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, !HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_5));
        // printf("程序运行中！\r\n");
    }
}

/**
 * 横
 */
void GPIO_Horizontal(bool level)
{
    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    if (level)
    { // 输入上拉
        // printf("横->输入上拉\n");
        GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
        GPIO_InitStruct.Pull = GPIO_PULLUP;
    }
    else
    { // 输出低电平
        // printf("横->输出低电平\n");
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
    }
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct); // GPIO初始化

    GPIO_InitStruct.Pin = GPIO_PIN_10 | GPIO_PIN_5;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct); // GPIO初始化

    if (!level)
    {
        // printf("横->设置低电平\n");
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9, 0);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, 0);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 0);
    }
}

#define pinddddd GPIO_PIN_9

/**
 *竖
 */
void GPIO_Vertical(bool level)
{
    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_10;
    if (level)
    { // 输入上拉
        // printf("竖->输入上拉\n");
        GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
        GPIO_InitStruct.Pull = GPIO_PULLUP;
    }
    else
    { // 输出低电平
        // printf("竖->输出低电平\n");
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
    }

    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct); // GPIO初始化

    GPIO_InitStruct.Pin = GPIO_PIN_3 | GPIO_PIN_4;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct); // GPIO初始化

    if (!level)
    {
        // printf("竖->设置低电平\n");
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, 0);
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_10, 0);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_3, 0);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 0);
    }
}

void Key_Identify(uint8_t a, uint8_t b)
{
    if (a > 4 || b > 3)
        return;
    a -= 1;
    b -= 1;
    static const char KEY_Data[4][3] = {{'1', '2', '3'}, {'4', '5', '6'}, {'7', '8', '9'}, {'*', '0', '#'}};
    printf("val:%c,time:%ld\r\n", KEY_Data[a][b], HAL_GetTick());
}

void Vertical_Get(uint8_t Horizontal)
{
    GPIO_Horizontal(0);
    GPIO_Vertical(1);

    uint8_t Vertical = 0xff;

    if (!HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_8) || !HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_10) || !HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_3) || !HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_4))
    {
        if (!HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_8))
        {
            printf("竖1\n");
            Vertical = 1;
        }
        else if (!HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_10))
        {
            printf("竖2\n");
            Vertical = 2;
        }
        else if (!HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_3))
        {
            printf("竖3\n");
            Vertical = 3;
        }
        else if (!HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_4))
        {
            printf("竖4\n");
            Vertical = 4;
        }
        Key_Identify(Vertical, Horizontal);
    }
}

void Get_GPIO_Level_Task()
{

    /**
     * 引脚对应
     * PA9->1
     * PA8->2
     * PB10->3
     * PB4->4
     * PB5->5
     * PB3->6
     * PA10->7
     */
    /**
     * 键盘横
     * PB10->3
     * PA9->1
     * PB5->5
     */
    /**
     * 键盘竖
     * PA8->2
     * PA10->7
     * PB3->6
     * PB4->4
     */

    /**
     * 键盘横
     * PB10->x3
     * PA9->x2
     * PB5->x1
     * 
     * A9
     * A8
     * B10
     */
    /**
     * 键盘竖
     * PA8->y1
     * PA10->y2
     * PB3->y3
     * PB4->y4
     * 
     * B4
     * B5
     * A3
     * B10
     */
    static uint32_t Time = 0;
    if (HAL_GetTick() - Time >= 300)
    {
        Time = HAL_GetTick();
        GPIO_Horizontal(1);
        GPIO_Vertical(0);

        if (!HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_9) || !HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_5) || !HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_10))
        {
            HAL_Delay(20);
            if (!HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_9))
            {
                printf("横2\n");
                Vertical_Get(2);
            }
            else if (!HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_5))
            {
                printf("横3\n");
                Vertical_Get(3);
            }
            else if (!HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_10))
            {
                printf("横1\n");
                Vertical_Get(1);
            }
        }
        // printf("\r\n\r\n");
    }
}

int main()
{
    HAL_Init();
    LED_GPIO_Init();
    UART_init();
    __HAL_RCC_GPIOA_CLK_ENABLE(); // GPIO A 时钟初始化
    __HAL_RCC_GPIOB_CLK_ENABLE(); // GPIO B 时钟初始化

    printf("程序启动！\n程序启动！\n程序启动！\n程序启动！\n");
    for (;;)
    {
        LED_Task();
        Get_GPIO_Level_Task();
    }
}
