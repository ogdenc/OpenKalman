/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref values::internal::make_complex_number function.
 */

#ifndef OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP
#define OPENKALMAN_MAKE_COMPLEX_NUMBER_HPP

#include "values/interface/number_traits.hpp"
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/concepts/complex.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/traits/fixed_value_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/functions/to_value_type.hpp"
#include "values/classes/fixed_value.hpp"

namespace OpenKalman::values::internal
{
  namespace detail
  {
    /**
     * \internal
     * \brief A callable object that makes a \ref complex number associated with type T.
     * \tparam T The complex or real type of the result (or void if this type is to be derived)
     */
#ifdef __cpp_concepts
    template<typename T = void> requires number<T> or std::same_as<T, void>
#else
    template<typename T = void, typename = void>
#endif
    struct make_complex_number;


    /**
     * \internal
     * \brief A callable object that makes a \ref complex number associated with type T.
     * \tparam T The complex or real type of the result
     */
#ifdef __cpp_concepts
    template<number T>
    struct make_complex_number<T>
#else
    template<typename T>
    struct make_complex_number<T, std::enable_if_t<number<T>>>
#endif
    {
#if __cpp_nontype_template_args < 201911L
    private:

      template<typename C, typename Re, typename Im>
      struct FixedComplex
      {
        using value_type = C;
        static constexpr value_type value {interface::number_traits<C>::make_complex(fixed_value_of_v<Re>, fixed_value_of_v<Im>)};
        using type = FixedComplex;
        constexpr operator value_type() const { return value; }
        constexpr value_type operator()() const { return value; }
      };

    public:
#endif

      /**
       * \internal
       * \brief Make a complex number of type T from real and imaginary parts.
       * \tparam T The complex or real type of the result
       * \param re The real part.
       * \param im The imaginary part.
       * \return A \ref complex \ref value
       */
#ifdef __cpp_concepts
      template<value Re, value Im = fixed_value<real_type_of_t<T>, 0>> requires (not values::complex<Re>) and (not values::complex<Im>) and
        std::convertible_to<Re, real_type_of_t<T>> and std::convertible_to<Im, real_type_of_t<T>> and
        requires { interface::number_traits<std::decay_t<T>>::make_complex(to_value_type(std::declval<Re>()), to_value_type(std::declval<Im>())); }
      constexpr complex decltype(auto)
#else
      template<typename Re, typename Im = fixed_value<real_type_of_t<T>, 0>, std::enable_if_t<value<Re> and value<Im> and
        (not values::complex<Re>) and (not values::complex<Im>) and
        stdex::convertible_to<Re, real_type_of_t<T>> and stdex::convertible_to<Im, real_type_of_t<T>>, int> = 0>
      constexpr decltype(auto)
#endif
      operator()(Re&& re, Im&& im = {}) const
      {
        if constexpr (fixed<Re> and fixed<Im>)
        {
          constexpr auto r = fixed_value_of_v<Re>;
          constexpr auto i = fixed_value_of_v<Im>;
          using C = std::decay_t<decltype(interface::number_traits<std::decay_t<T>>::make_complex(r, i))>;
#if __cpp_nontype_template_args >= 201911L
          return fixed_value<C, r, i>{};
#else
          if constexpr (r == static_cast<std::intmax_t>(r) and i == static_cast<std::intmax_t>(i))
            return fixed_value<C, static_cast<std::intmax_t>(r), static_cast<std::intmax_t>(i)>{};
          else
            return FixedComplex<C, std::decay_t<Re>, std::decay_t<Im>>{};
#endif
        }
        else
        {
          return interface::number_traits<std::decay_t<T>>::make_complex(to_value_type(std::forward<Re>(re)), to_value_type(std::forward<Im>(im)));
        }
      }


      /**
       * \overload
       * \brief Convert a complex number of one real type from a complex number of another real type.
       * \tparam Arg A complex number to be converted.
       */
#ifdef __cpp_concepts
      template<complex Arg> requires std::convertible_to<real_type_of_t<Arg>, real_type_of_t<T>>
      constexpr complex decltype(auto)
#else
      template<typename Arg, std::enable_if_t<number<T> and complex<Arg> and
        stdex::convertible_to<real_type_of_t<Arg>, real_type_of_t<T>>, int> = 0>
      constexpr decltype(auto)
#endif
      operator()(Arg&& arg) const
      {
        if constexpr (std::is_same_v<real_type_of_t<T>, real_type_of_t<Arg>>)
        {
          return std::forward<Arg>(arg);
        }
        else
        {
          return operator()(values::real(std::forward<Arg>(arg)), values::imag(std::forward<Arg>(arg)));
        }
      }
    };


    /**
     * \internal
     * \brief A callable object that makes a \ref complex number associated with type T.
     * \tparam T The complex or real type of the result
     */
    template<>
    struct make_complex_number<void>
    {
      /**
       * \brief Make a complex number from real and imaginary parts, deriving the complex type from the arguments.
       * \param re The real part.
       * \param im The imaginary part.
       */
#ifdef __cpp_concepts
      template<number Re, number Im = fixed_value<real_type_of_t<Re>, 0>> requires
        (not complex<Re>) and (not complex<Im>) and std::common_with<value_type_of_t<Re>, value_type_of_t<Im>>
      constexpr complex decltype(auto)
#else
      template<typename Re, typename Im = fixed_value<real_type_of_t<Re>, 0>, std::enable_if_t<
        number<Re> and number<Im> and (not complex<Re>) and (not complex<Im>), int> = 0>
      constexpr decltype(auto)
#endif
      operator()(Re&& re, Im&& im) const
      {
        using T = std::decay_t<std::common_type_t<value_type_of_t<Re>, value_type_of_t<Im>>>;
        return make_complex_number<T>{}(std::forward<Re>(re), std::forward<Im>(im));
      }
    };
  }


#ifdef __cpp_concepts
  template<typename T = void> requires number<T> or std::same_as<T, void>
#else
  template<typename T = void, typename = void>
#endif
  inline constexpr auto make_complex_number = detail::make_complex_number<T>{};


}

#endif
