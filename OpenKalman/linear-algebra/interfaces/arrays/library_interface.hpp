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
 * \brief Definition of library_interface for C++ arrays.
 */

#ifndef OPENKALMAN_INTERFACES_ARRAYS_LIBRARY_INTERFACE_HPP
#define OPENKALMAN_INTERFACES_ARRAYS_LIBRARY_INTERFACE_HPP

#include <algorithm>
#include "linear-algebra/interfaces/default/library_interface.hpp"
#include "linear-algebra/concepts/writable.hpp"
#include "linear-algebra/concepts/has_untyped_index.hpp"
#include "linear-algebra/functions/get_component.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to standard c++ arrays of any rank.
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_array_v<T>
  struct library_interface<T>
#else
  template<typename T>
  struct library_interface<T, std::enable_if_t<std::is_array_v<T>>>
#endif
  {
    // library_base not defined

  private:

      template<std::size_t rank = 0, typename U, typename Indices>
      static constexpr decltype(auto)
      get_component_impl(U&& u, const Indices& indices)
      {
        if constexpr (rank < std::rank_v<T>)
        {
          auto i = values::to_value_type(collections::get(indices, std::integral_constant<std::size_t, rank>{}));
          return get_component_impl<rank + 1>(std::forward<U>(u)[i], indices);
        }
        else
        {
          return std::forward<U>(u);
        }
      }

  public:

    /**
     * \brief Get the element based on row-major indices.
     */
    static constexpr auto
    get_component = [](auto&& t, const auto& indices) -> decltype(auto)
    {
      return get_component_impl(std::forward<decltype(t)>(t), indices);
    };

  private:

      template<std::size_t rank = 0, typename U, typename Indices, typename S>
      static constexpr void
      set_component_impl(U& u, const Indices& indices, const S& s)
      {
        if constexpr (rank < std::rank_v<T>)
        {
          auto i = values::to_value_type(collections::get(indices, std::integral_constant<std::size_t, rank>{}));
          set_component_impl<rank + 1>(u[i], indices, s);
        }
        else
        {
          u = s;
        }
      }

  public:

    /**
     * \brief Set a component based on row-major indices.
     */
    static constexpr auto
    set_component = [](T& t, const auto& indices, const std::remove_all_extents_t<T>& s) -> void
    {
      set_component_impl(t, indices, s);
    };


    /**
     * \brief This is effectively an identity function. \todo do we need this?
     */
    static constexpr auto
    to_native_matrix = [](auto&& t) -> decltype(auto)
    {
      return std::forward<decltype(t)>(t);
    };


    // assign is not defined
    // make_default is not defined because the array could only be returned as a pointer to an array made in the heap
    // fill_components is not defined. This will be handled by the global function.
    // make_constant is not defined
    // make_identity_matrix is not defined.
    // make_triangular_matrix is not defined.
    // make_hermitian_adapter is not defined.


     /**
      * \brief Project the vector space associated with index 0 to a Euclidean space for applying directional statistics.
      * \note This is optional.
      * If not defined, the \ref OpenKalman::to_euclidean "to_euclidean" function will construct a ToEuclideanExpr object.
      * In this case, the library should be able to accept the ToEuclideanExpr object as native.
      */
    static constexpr auto
#ifdef __cpp_concepts
    to_euclidean = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> has_untyped_index<0> decltype(auto)
#else
    to_euclidean = [](auto&& arg) -> decltype(auto)
#endif
    {
      // ...
    };


     /**
      * \brief Project the Euclidean vector space associated with index 0 to \ref coordinates::pattern v after applying directional statistics
      * \param v The new \ref coordinates::pattern for index 0.
      * \note This is optional.
      * If not defined, the \ref OpenKalman::from_euclidean "from_euclidean" function will construct a FromEuclideanExpr object.
      * In this case, the library should be able to accept FromEuclideanExpr object as native.
      */
 #ifdef __cpp_concepts
     template<indexible Arg, coordinates::pattern V>
     static constexpr indexible auto
 #else
     template<typename Arg, typename V>
     static constexpr auto
 #endif
     from_euclidean(Arg&& arg, const V& v) = delete;


     /**
      * \brief Wrap Arg based on \ref coordinates::pattern V.
      * \note This is optional. If not defined, the public \ref OpenKalman::wrap_angles "wrap_angles" function
      * will call <code>from_euclidean(to_euclidean(std::forward<Arg>(arg)), get_pattern_collection<0>(arg))</code>.
      */
 #ifdef __cpp_concepts
     template<indexible Arg>
     static constexpr indexible auto
 #else
     template<typename Arg>
     static constexpr auto
 #endif
     wrap_angles(Arg&& arg) = delete;


    /**
     * \brief Get a block from a matrix or tensor.
     * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref values::index.
     * \param size A tuple corresponding to each of indices, each element specifying the size (as an \ref values::index) of the extracted block.
     */
#ifdef __cpp_concepts
    template<indexible Arg, values::index...Begin, values::index...Size> requires
      (index_count_v<Arg> == sizeof...(Begin)) and (index_count_v<Arg> == sizeof...(Size))
    static indexible decltype(auto)
#else
    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto)
#endif
    get_slice(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size) = delete;


    /**
     * \brief Set a block from a \ref writable matrix or tensor.
     * \param arg The matrix or tensor to be modified.
     * \param block A block to be copied into Arg at a particular location.
     * Note: This may not necessarily be within the same library as arg, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \param begin \ref values::index corresponding to each of indices, specifying the beginning \ref values::index.
     * \returns An lvalue reference to arg.
     */
#ifdef __cpp_concepts
    template<writable Arg, indexible Block, values::index...Begin> requires
      (index_count_v<Block> == sizeof...(Begin)) and (index_count_v<Arg> == sizeof...(Begin))
#else
    template<typename Arg, typename Block, typename...Begin>
#endif
    static void
    set_slice(Arg& arg, Block&& block, const Begin&...begin) = delete;


    // set_triangle is not defined.
    // to_diagonal is not defined.
    // diagonal_of is not defined
    // broadcast is not defined.
    // n_ary_operation is not defined.
    // reduce is not defined
    // conjugate is not defined.
    // transpose is not defined.
    // adjoint is not defined.
    // determinant is not defined.
    // sum is not defined.
    // scalar_product is not defined.
    // scalar_quotient is not defined.
    // contract is not defined.
    // contract_in_place is not defined.
    // cholesky_factor is not defined.
    // rank_update_hermitian is not defined.
    // rank_update_triangular is not defined.
    // solve is not defined.
    // LQ_decomposition is not defined.
    // QR_decomposition is not defined.

  };


}


#endif
