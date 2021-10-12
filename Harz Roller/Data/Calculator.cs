using System;
using System.Collections.Generic;
using System.Text;

namespace Harz_Roller.Data
{
    class Calculator
    {
        float[][] dataPlane;
        int pointer;

        public Calculator(float[][] dataPlane, int pointer = 0)
        {
            this.dataPlane = dataPlane;
            this.pointer = pointer;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="timeFrames">must be ascending order sizes of each batch to check for if it is a peak</param>
        /// <returns></returns>
        public bool[] isPeak(int[] timeFrames, int column, bool step = true)
        {
            bool[] temp = new bool[timeFrames.Length];
            int currentFrame = 0;

            for(int i = 0; i <= timeFrames[timeFrames.Length-1]; i++)
            {
                if (dataPlane[pointer + i][column] > dataPlane[pointer][column]) break;
                if (i == timeFrames[currentFrame])
                {
                    temp[currentFrame] = true;
                    currentFrame++;
                }
            }

            if (step) pointer++;
            return temp;
        }
        public bool[] isValley(int[] timeFrames, int column, bool step = true)
        {
            bool[] temp = new bool[timeFrames.Length];
            int currentFrame = 0;

            for (int i = 0; i <= timeFrames[timeFrames.Length - 1]; i++)
            {
                if (dataPlane[pointer + i][column] < dataPlane[pointer][column]) break;
                if (i == timeFrames[currentFrame])
                {
                    temp[currentFrame] = true;
                    currentFrame++;
                }
            }

            if (step) pointer++;
            return temp;
        }

        public void makeDirectional(int column)
        {
            for(int i = dataPlane.GetLength(0)-1; i > 0 ; i--)
            {
                dataPlane[i][column] = (dataPlane[i][column] / dataPlane[i - 1][column])-1;
            }
        }
        public void setPointer(int pointer = 0)
        {
            this.pointer = pointer;
        }

        public float[][] getDataPlane()
        {
            return dataPlane;
        }
    }
}
