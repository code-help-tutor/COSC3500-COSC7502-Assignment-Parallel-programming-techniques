WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
#define RUBRIC_CPU 0
#define RUBRIC_GPU 1
#define RUBRIC_MPI 2
#define RUBRIC_LENGTH 8

float rubric[3][RUBRIC_LENGTH];

int rubricInit()
{
    //CPU MARKS
    rubric[RUBRIC_CPU][7] = 2.3;
    rubric[RUBRIC_CPU][6] = 16.5;
    rubric[RUBRIC_CPU][5] = 36.2;
    rubric[RUBRIC_CPU][4] = 64.3;
    rubric[RUBRIC_CPU][3] = 121.0;
    rubric[RUBRIC_CPU][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_CPU][1] = 1000;
    rubric[RUBRIC_CPU][0] = 1000;

    //GPU MARKS
    rubric[RUBRIC_GPU][7] = 4.9;
    rubric[RUBRIC_GPU][6] = 8;
    rubric[RUBRIC_GPU][5] = 24.8;
    rubric[RUBRIC_GPU][4] = 45.7;
    rubric[RUBRIC_GPU][3] = 100;
    rubric[RUBRIC_GPU][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_GPU][1] = 1000;
    rubric[RUBRIC_GPU][0] = 1000;

    //MPI MARKS
    rubric[RUBRIC_MPI][7] = 1.4;
    rubric[RUBRIC_MPI][6] = 8.3;
    rubric[RUBRIC_MPI][5] = 18.5;
    rubric[RUBRIC_MPI][4] = 31.6;
    rubric[RUBRIC_MPI][3] = 60.8;
    rubric[RUBRIC_MPI][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_MPI][1] = 1000;
    rubric[RUBRIC_MPI][0] = 1000;
    return 1;
}

float getGrade(float performanceFactor, float err, float* gradeTable)
{
    //floating point error tolerance of the answer given
    const float errTolerance = 1e-7;

    if (abs(err) <= errTolerance)
    {
        // Matrix multiplication works, but is about as slow as possible.
        if (performanceFactor >= gradeTable[2])
        {
            return 2;
        }
        else
        {
            // God-like performance. Full marks.
            if (performanceFactor < gradeTable[RUBRIC_LENGTH - 1])
            {
                return RUBRIC_LENGTH - 1;
            }
            else
            {
                for (int gradeIdx = RUBRIC_LENGTH; gradeIdx >= 1; gradeIdx--)
                {
                    // Linearly interpolate between the levels in the grade table to assign a grade
                    if (performanceFactor > gradeTable[gradeIdx] && performanceFactor <= gradeTable[gradeIdx - 1])
                    {
                        float x1 = gradeTable[gradeIdx];
                        float x2 = gradeTable[gradeIdx - 1];
                        float y1 = gradeIdx;
                        float y2 = gradeIdx - 1;
                        float x = performanceFactor;

                        float grade = y1 + ((x - x1) / (x2 - x1) * (y2 - y1));
                        grade = floor(grade * 10.0);
                        grade = grade / 10.0;
                        return grade;
                    }
                }
            }
        }
    }
    else
    {
        // Matrix multiplication doesn't work properly. 1 point for submitting something that runs at least.
        return 1;
    }
    return 1;
}