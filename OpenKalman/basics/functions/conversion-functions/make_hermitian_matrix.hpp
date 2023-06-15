/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_hermitian_matrix.
 */

#ifndef OPENKALMAN_MAKE_HERMITIAN_MATRIX_HPP
#define OPENKALMAN_MAKE_HERMITIAN_MATRIX_HPP

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, HermitianAdapterType t, typename = void>
    struct make_hermitian_adapter_defined: std::false_type {};

    template<typename T, HermitianAdapterType t>
    struct make_hermitian_adapter_defined<T, t, std::void_t<
      decltype(interface::HermitianTraits<std::decay_t<T>>::template make_hermitian_adapter<t>(std::declval<T&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Wraps a matrix in a \ref hermitian_adapter to create a \ref hermitian_matrix.
   * \note The resulting adapter type is not guaranteed to be an adapter or have the requested adapter_type.
   * \tparam adapter_type The intended \ref TriangleType of the result (lower op upper).
   * \tparam Arg A non-hermitian matrix.
   */
#ifdef __cpp_concepts
  template<HermitianAdapterType adapter_type = HermitianAdapterType::lower, square_matrix<Likelihood::maybe> Arg>
    requires (not hermitian_matrix<Arg>) and (adapter_type == HermitianAdapterType::lower or adapter_type == HermitianAdapterType::upper)
  constexpr hermitian_matrix decltype(auto)
#else
  template<HermitianAdapterType adapter_type = HermitianAdapterType::lower, typename Arg, std::enable_if_t<
    square_matrix<Arg, Likelihood::maybe> and (not hermitian_matrix<Arg>) and
    (adapter_type == HermitianAdapterType::lower or adapter_type == HermitianAdapterType::upper), int> = 0>
  constexpr decltype(auto)
#endif
  make_hermitian_matrix(Arg&& arg)
  {
    using Traits = interface::HermitianTraits<std::decay_t<Arg>>;
    using pArg = std::conditional_t<std::is_lvalue_reference_v<Arg>, Arg, std::remove_reference_t<decltype(make_self_contained(arg))>>;

    if constexpr (hermitian_matrix<Arg, Likelihood::maybe>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::invalid_argument {"Argument to make_hermitian_matrix is not a square matrix."};

      if constexpr (hermitian_matrix<Arg>)
        return std::forward<Arg>(arg);
      else if constexpr (hermitian_adapter<Arg, adapter_type>)
        return make_hermitian_matrix<adapter_type>(nested_matrix(std::forward<Arg>(arg)));
      else if constexpr (hermitian_adapter<Arg>)
      {
        constexpr auto t = adapter_type == HermitianAdapterType::lower ? HermitianAdapterType::upper : HermitianAdapterType::lower;
        return make_hermitian_matrix<t>(nested_matrix(std::forward<Arg>(arg)));
      }
      else
        return SelfAdjointMatrix<pArg, adapter_type> {std::forward<Arg>(arg)};
    }
# ifdef __cpp_concepts
    else if constexpr (requires (Arg&& arg) { Traits::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg)); })
# else
    else if constexpr (detail::make_hermitian_adapter_defined<Arg, adapter_type>::value)
# endif
    {
      auto new_h {Traits::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg))};
      static_assert(hermitian_matrix<decltype(new_h), Likelihood::maybe>, "make_hermitian_matrix interface must return a hermitian matrix");
      if constexpr (hermitian_matrix<decltype(new_h)>) return new_h;
      else return make_hermitian_matrix<adapter_type>(std::move(new_h));
    }
    else
    {
      // Default behavior if interface function not defined:
      return SelfAdjointMatrix<pArg, adapter_type> {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_HERMITIAN_MATRIX_HPP
