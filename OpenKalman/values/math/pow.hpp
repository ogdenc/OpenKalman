/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \brief Definition for \ref values::pow.
 */

#ifndef OPENKALMAN_VALUE_POW_HPP
#define OPENKALMAN_VALUE_POW_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"
#include "values/math/copysign.hpp"
#include "values/math/exp.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::values
{
  namespace detail
  {
    template<typename Arg, typename N>
    constexpr Arg pow_integral(const Arg& arg, const N& n)
    {
      if (n == 0)
      {
        return Arg{1};
      }
      else if (arg != Arg{0} and n < 0)
      {
        return Arg{1} / pow_integral(arg, -n);
      }
      else if (n % 2 == 1) // positive odd
      {
        if (arg == Arg{0})
          return arg;
        else
          return arg * pow_integral(arg, n - 1);
      }
      else if (-n % 2 == 1) // negative odd
      {
        if (arg == Arg{0})
          return Arg {values::copysign(values::internal::infinity<values::real_type_of_t<Arg>>(), values::real(arg))};
        else
          return pow_integral(arg, n + 1) / arg;
      }
      else // positive even or negative even
      {
        if (arg == Arg{0})
          return n > 0 ? static_cast<Arg>(+0.) : Arg {+values::internal::infinity<values::real_type_of_t<Arg>>()};
        else
          return pow_integral(arg, n / 2) * pow_integral(arg, n / 2);
      }
    }
  }


  /**
   * \internal
   * \brief A constexpr alternative to the std::pow function.
   * \param x The operand
   * \param n The power
   * \return x to the power of n.
   */
#ifdef __cpp_concepts
  template<values::value Arg, values::value Exponent> requires
    (values::integral<Exponent> or std::common_with<number_type_of_t<Arg>, number_type_of_t<Exponent>>)
  constexpr values::value auto pow(const Arg& arg, const Exponent& exponent)
#else
  template <typename Arg, typename Exponent, std::enable_if_t<values::value<Arg> and values::value<Exponent> and
    (values::integral<Exponent> or
      std::is_void_v<std::void_t<typename std::common_type<number_type_of_t<Arg>, number_type_of_t<Exponent>>::type>>), int> = 0>
  constexpr auto pow(const Arg& arg, const Exponent& exponent)
#endif
  {
    if constexpr (not values::number<Arg> or not values::number<Exponent>)
    {
      struct Op
      {
        using NA = values::number_type_of_t<Arg>;
        using NE = values::number_type_of_t<Exponent>;
        constexpr auto operator()(const NA& a, const NE& e) const { return values::pow(a, e); }
      };
      return values::operation {Op{}, arg, exponent};
    }
    else
    {
      using std::pow;
      using Return = decltype(pow(arg, exponent));
      using R = decltype(values::real(arg));
      struct Op { auto operator()(const Arg& a, const Exponent& e) { return pow(a, e); } };
      if (internal::constexpr_callable<Op>(arg, exponent)) return pow(arg, exponent);
      else if constexpr (values::integral<Exponent>)
      {
        if constexpr (values::complex<Arg>)
        {
          // Errors should be the same as if the exponent were non-integral:
          if (values::isinf(values::real(arg)) or values::isinf(values::imag(arg)) or values::isnan(values::real(arg)) or values::isnan(values::imag(arg)))
            return values::pow(arg, values::real(exponent));
          else
            return internal::make_complex_number<Return>(detail::pow_integral(internal::make_complex_number<R>(arg), exponent));
        }
        else
        {
          if (arg == 1) return Return{1};
          if (exponent == 0) return Return{1};
          if (values::isnan(arg)) return values::internal::NaN<Return>();
          if constexpr (std::numeric_limits<Arg>::has_infinity)
          {
            if (exponent % 2 == 1) // positive odd
            {
              if (arg == -std::numeric_limits<Arg>::infinity()) return -values::internal::infinity<Return>();
              else if (arg == +std::numeric_limits<Arg>::infinity()) return +values::internal::infinity<Return>();
            }
            else if (-exponent % 2 == 1) // negative odd
            {
              if (arg == -std::numeric_limits<Arg>::infinity()) return static_cast<Return>(-0.);
              else if (arg == +std::numeric_limits<Arg>::infinity()) return static_cast<Return>(+0.);
            }
            else if (arg == -std::numeric_limits<Arg>::infinity() or arg == +std::numeric_limits<Arg>::infinity())
            {
              if (exponent > 0) return +values::internal::infinity<Return>(); // positive even
              else return static_cast<Return>(+0.); // negative even
            }
          }
          return detail::pow_integral(values::real(arg), exponent);
        }
      }
      else
      {
        if constexpr (values::complex<Return>)
        {
          auto lg = values::log(internal::make_complex_number<Return>(arg));
          if constexpr (values::complex<Exponent>)
          {
            auto a = values::real(lg);
            auto b = values::imag(lg);
            auto c = values::real(exponent);
            auto d = values::imag(exponent);
            auto lge = internal::make_complex_number<R>(a*c - b*d, a*d + b*c);
            return internal::make_complex_number<Return>(values::exp(lge));
          }
          else
          {
            auto lge = internal::make_complex_number<R>(values::real(lg) * exponent, values::imag(lg) * exponent);
            return internal::make_complex_number<Return>(values::exp(lge));
          }
        }
        else
        {
          if (arg == 1) return Return{1};
          if constexpr (std::numeric_limits<Arg>::has_infinity)
          {
            if (values::isinf(arg))
            {
              if (exponent < 0) return static_cast<Return>(+0.);
              else return values::internal::infinity<Return>();
            }
            else if (exponent == -std::numeric_limits<Exponent>::infinity())
            {
              if (-1 < arg and arg < 1) return +values::internal::infinity<Return>();
              else if (arg < -1 or 1 < arg) return static_cast<Return>(+0.);
              else return Return{1}; // x == -1 (x == 1 case handled above)
            }
            else if (exponent == +std::numeric_limits<Exponent>::infinity())
            {
              if (-1 < arg and arg < 1) return static_cast<Return>(+0.);
              else if (arg < -1 or 1 < arg) return +values::internal::infinity<Return>();
              else return Return{1}; // x == -1 (x == 1 case handled above)
            }
          }

          if (arg > 0)
          {
            if (exponent == 1) return static_cast<Return>(arg);
            else return static_cast<Return>(values::exp(values::log(arg) * exponent));
          }
          else if (arg == 0)
          {
            if (exponent < 0) return +values::internal::infinity<Return>();
            else return static_cast<Return>(+0.);
          }
          else return values::internal::NaN<Return>();
        }
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_POW_HPP
