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
 * \brief Definition for \ref value::pow.
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

namespace OpenKalman::value
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
          return Arg {value::copysign(value::internal::infinity<value::real_type_of_t<Arg>>(), value::real(arg))};
        else
          return pow_integral(arg, n + 1) / arg;
      }
      else // positive even or negative even
      {
        if (arg == Arg{0})
          return n > 0 ? static_cast<Arg>(+0.) : Arg {+value::internal::infinity<value::real_type_of_t<Arg>>()};
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
  template<value::value Arg, value::value Exponent> requires
    (value::integral<Exponent> or std::common_with<number_type_of_t<Arg>, number_type_of_t<Exponent>>)
  constexpr value::value auto pow(const Arg& arg, const Exponent& exponent)
#else
  template <typename Arg, typename Exponent, std::enable_if_t<value::value<Arg> and value::value<Exponent> and
    (value::integral<Exponent> or
      std::is_void_v<std::void_t<typename std::common_type<number_type_of_t<Arg>, number_type_of_t<Exponent>>::type>>), int> = 0>
  constexpr auto pow(const Arg& arg, const Exponent& exponent)
#endif
  {
    if constexpr (not value::number<Arg> or not value::number<Exponent>)
    {
      struct Op
      {
        using NA = value::number_type_of_t<Arg>;
        using NE = value::number_type_of_t<Exponent>;
        constexpr auto operator()(const NA& a, const NE& e) const { return value::pow(a, e); }
      };
      return value::operation {Op{}, arg, exponent};
    }
    else
    {
      using std::pow;
      using Return = decltype(pow(arg, exponent));
      using R = decltype(value::real(arg));
      struct Op { auto operator()(const Arg& a, const Exponent& e) { return pow(a, e); } };
      if (internal::constexpr_callable<Op>(arg, exponent)) return pow(arg, exponent);
      else if constexpr (value::integral<Exponent>)
      {
        if constexpr (value::complex<Arg>)
        {
          // Errors should be the same as if the exponent were non-integral:
          if (value::isinf(value::real(arg)) or value::isinf(value::imag(arg)) or value::isnan(value::real(arg)) or value::isnan(value::imag(arg)))
            return value::pow(arg, value::real(exponent));
          else
            return internal::make_complex_number<Return>(detail::pow_integral(internal::make_complex_number<R>(arg), exponent));
        }
        else
        {
          if (arg == 1) return Return{1};
          if (exponent == 0) return Return{1};
          if (value::isnan(arg)) return value::internal::NaN<Return>();
          if constexpr (std::numeric_limits<Arg>::has_infinity)
          {
            if (exponent % 2 == 1) // positive odd
            {
              if (arg == -std::numeric_limits<Arg>::infinity()) return -value::internal::infinity<Return>();
              else if (arg == +std::numeric_limits<Arg>::infinity()) return +value::internal::infinity<Return>();
            }
            else if (-exponent % 2 == 1) // negative odd
            {
              if (arg == -std::numeric_limits<Arg>::infinity()) return static_cast<Return>(-0.);
              else if (arg == +std::numeric_limits<Arg>::infinity()) return static_cast<Return>(+0.);
            }
            else if (arg == -std::numeric_limits<Arg>::infinity() or arg == +std::numeric_limits<Arg>::infinity())
            {
              if (exponent > 0) return +value::internal::infinity<Return>(); // positive even
              else return static_cast<Return>(+0.); // negative even
            }
          }
          return detail::pow_integral(value::real(arg), exponent);
        }
      }
      else
      {
        if constexpr (value::complex<Return>)
        {
          auto lg = value::log(internal::make_complex_number<Return>(arg));
          if constexpr (value::complex<Exponent>)
          {
            auto a = value::real(lg);
            auto b = value::imag(lg);
            auto c = value::real(exponent);
            auto d = value::imag(exponent);
            auto lge = internal::make_complex_number<R>(a*c - b*d, a*d + b*c);
            return internal::make_complex_number<Return>(value::exp(lge));
          }
          else
          {
            auto lge = internal::make_complex_number<R>(value::real(lg) * exponent, value::imag(lg) * exponent);
            return internal::make_complex_number<Return>(value::exp(lge));
          }
        }
        else
        {
          if (arg == 1) return Return{1};
          if constexpr (std::numeric_limits<Arg>::has_infinity)
          {
            if (value::isinf(arg))
            {
              if (exponent < 0) return static_cast<Return>(+0.);
              else return value::internal::infinity<Return>();
            }
            else if (exponent == -std::numeric_limits<Exponent>::infinity())
            {
              if (-1 < arg and arg < 1) return +value::internal::infinity<Return>();
              else if (arg < -1 or 1 < arg) return static_cast<Return>(+0.);
              else return Return{1}; // x == -1 (x == 1 case handled above)
            }
            else if (exponent == +std::numeric_limits<Exponent>::infinity())
            {
              if (-1 < arg and arg < 1) return static_cast<Return>(+0.);
              else if (arg < -1 or 1 < arg) return +value::internal::infinity<Return>();
              else return Return{1}; // x == -1 (x == 1 case handled above)
            }
          }

          if (arg > 0)
          {
            if (exponent == 1) return static_cast<Return>(arg);
            else return static_cast<Return>(value::exp(value::log(arg) * exponent));
          }
          else if (arg == 0)
          {
            if (exponent < 0) return +value::internal::infinity<Return>();
            else return static_cast<Return>(+0.);
          }
          else return value::internal::NaN<Return>();
        }
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_POW_HPP
