#ifndef CONVERT_HPP
#define CONVERT_HPP

namespace libtorchPG
{
    static float toFloat(int value)
    {
        return static_cast<float>(value);
    }

    static int toInt(float value)
    {
        return static_cast<int>(value);
    }
} // namespace libtorchPG

#endif