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
#include "linear-algebra/traits/scalar_type_of.hpp"
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

      template<std::size_t rank = 0>
      static constexpr decltype(auto)
      get_component_impl(auto&& t, const auto& indices)
      {
        if constexpr (rank == index_count_v<T>)
        {
          return std::forward<decltype(t)>(t);
        }
        else
        {
          auto i = values::to_value_type(collections::get(indices, std::integral_constant<std::size_t, rank>{}));
          return get_component_impl<rank + 1>(std::forward<decltype(t)>(t)[i], indices);
        }
      }

  public:

    /**
     * \brief Get the element based on row-major indices.
     */
    static constexpr auto
    get_component = [](auto&& t, const auto& indices)
    {
      return get_component_impl(std::forward<decltype(t)>(t), indices);
    };

  private:

      template<std::size_t rank = 0>
      static constexpr void
      set_component_impl(T& t, const auto& indices, const auto& s)
      {
        if constexpr (rank == index_count_v<T>)
        {
          t = s;
        }
        else
        {
          auto i = values::to_value_type(collections::get(indices, std::integral_constant<std::size_t, rank>{}));
          set_component_impl<rank + 1>(t[i], indices, s);
        }
      }

  public:

    /**
     * \brief Set a component based on row-major indices.
     */
    static constexpr auto
    set_component = [](T& t, const auto& indices, const scalar_type_of_t<T>& s) -> void
    {
      set_component_impl(t, indices, s);
    };


    /**
     * \brief Converts the argument to a native c++ arrayConverts Arg (if it is not already) to a native matrix operable within the library associated with LibraryObject.
     * \details The result should be in a form for which basic matrix operations can be performed within the library for LibraryObject.
     * This should be a lightweight transformation that does not require copying of elements.
     * If possible, properties such as \ref diagonal_matrix, \ref triangular_matrix, \ref hermitian_matrix,
     * \ref constant_matrix, and \ref constant_diagonal_matrix should be preserved in the resulting object.
     * \note if not defined, a call to \ref OpenKalman::to_native_matrix will construct a \ref LibraryWrapper.
     */
    static constexpr auto
    to_native_matrix = [](auto&& t)
    {
      return std::forward<decltype(t)>(t);
    };

  private:

      template<std::size_t rank = 0>
      static constexpr void
      assign_impl(T& t, auto&& other)
      {
        if constexpr (rank + 1 < index_count_v<T>)
        {
          set_component_impl<rank + 1>(t[i], indices, s);
        }
        else
        {
          std::copy(stdcompat::ranges::begin(other), stdcompat::ranges::end(other), stdcompat::ranges::begin(t));
        }
      }

  public:

    /**
     * \brief Copy or move into an array.
     */
    static constexpr auto
    assign = [](T& t, auto&& other) -> void
    {
      if constexpr (std::is_array_v<decltype(other)> and std::rank_v<decltype(other)> == std::rank_v<T> and stdcompat::assignable_from<>)
      {
        ;
      }
      t = std::forward<decltype(other)>(other);
    };


    /**
     * \brief Make a default, potentially uninitialized, dense, writable matrix or array within the library.
     * \details Takes a \ref coordinates::euclidean_pattern_collection that specifies the dimensions of the resulting object
     * \tparam layout the \ref data_layout of the result, which may be data_layout::left, data_layout::right, or
     * data_layout::none (which indicates the default layout for the library).
     * \tparam Scalar The scalar value of the result.
     * \return A default, potentially uninitialized, dense, writable object.
     * \note The interface may base the return value on any properties of LibraryObject (e.g., whether LibraryObject is a matrix or array).
     */
#ifdef __cpp_concepts
    template<data_layout layout, values::number Scalar> requires (layout != data_layout::stride)
    static auto
    make_default(coordinates::euclidean_pattern_collection auto&& descriptors) = delete;
#else
    template<data_layout layout, typename Scalar, typename Descriptors>
    static auto
    make_default(Descriptors&& descriptors) = delete;
#endif


    /**
     * \brief Fill a writable matrix with a list of elements.
     * \tparam Arg The \ref writable object to be filled.
     * \tparam layout The \ref data_layout of the listed elements, which may be \ref data_layout::left or \ref data_layout::right.
     * \param scalars A set of scalar values representing the elements. There must be exactly the right number of elements
     * to fill Arg.
     */
#ifdef __cpp_concepts
    template<data_layout layout, writable Arg> requires (layout == data_layout::right) or (layout == data_layout::left)
    static void
    fill_components(Arg& arg, const std::convertible_to<scalar_type_of_t<Arg>> auto ... scalars) = delete;
#else
    template<data_layout layout, typename Arg, typename...Scalars, std::enable_if_t<
      writable<Arg> and (layout == data_layout::right) or (layout == data_layout::left), int> = 0>
    static void
    fill_components(Arg& arg, const Scalars...scalars) = delete;
#endif


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
 #ifdef __cpp_concepts
     template<indexible Arg>
     static constexpr indexible auto
 #else
     template<typename Arg>
     static constexpr auto
 #endif
     to_euclidean(Arg&& arg) = delete;


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
