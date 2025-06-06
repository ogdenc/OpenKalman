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
  /**
   * \brief Creates a \ref hermitian_matrix by, if necessary, wrapping the argument in a \ref hermitian_adapter.
   * \note The result is guaranteed to be hermitian, but is not guaranteed to be an adapter or have the requested adapter_type.
   * \tparam adapter_type The intended \ref HermitianAdapterType of the result (lower op upper).
   * \tparam Arg A square matrix.
   */
#ifdef __cpp_concepts
  template<HermitianAdapterType adapter_type = HermitianAdapterType::lower, square_shaped<Applicability::permitted> Arg>
    requires (adapter_type == HermitianAdapterType::lower) or (adapter_type == HermitianAdapterType::upper)
  constexpr hermitian_matrix decltype(auto)
#else
  template<HermitianAdapterType adapter_type = HermitianAdapterType::lower, typename Arg, std::enable_if_t<
    square_shaped<Arg, Applicability::permitted> and
    (adapter_type == HermitianAdapterType::lower or adapter_type == HermitianAdapterType::upper), int> = 0>
  constexpr decltype(auto)
#endif
  make_hermitian_matrix(Arg&& arg)
  {
    constexpr auto transp = adapter_type == HermitianAdapterType::lower ? HermitianAdapterType::upper : HermitianAdapterType::lower;

    if constexpr (hermitian_matrix<Arg, Applicability::permitted>)
    {
      if constexpr (hermitian_matrix<Arg>)
        return std::forward<Arg>(arg);
      else if constexpr (hermitian_adapter<Arg, adapter_type>)
        return make_hermitian_matrix<adapter_type>(nested_object(std::forward<Arg>(arg)));
      else if constexpr (hermitian_adapter<Arg>)
        return make_hermitian_matrix<transp>(nested_object(std::forward<Arg>(arg)));
      else
        return HermitianAdapter<Arg, adapter_type> {std::forward<Arg>(arg)};
    }
    else if constexpr (triangular_adapter<Arg>)
    {
      if constexpr (triangular_matrix<Arg, static_cast<TriangleType>(adapter_type)>)
        return make_hermitian_matrix<adapter_type>(nested_object(std::forward<Arg>(arg)));
      else
        return make_hermitian_matrix<transp>(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (interface::make_hermitian_adapter_defined_for<Arg, adapter_type, Arg>)
    {
      using Traits = interface::library_interface<std::decay_t<Arg>>;
      auto new_h {Traits::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg))};
      static_assert(hermitian_matrix<decltype(new_h), Applicability::permitted>, "make_hermitian_matrix interface must return a hermitian matrix");
      if constexpr (hermitian_matrix<decltype(new_h)>) return new_h;
      else return make_hermitian_matrix<adapter_type>(std::move(new_h));
    }
    else
    {
      // Default behavior if interface function not defined:
      return HermitianAdapter<Arg, adapter_type> {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_HERMITIAN_MATRIX_HPP
