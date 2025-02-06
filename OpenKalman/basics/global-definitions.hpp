/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Global definitions for OpenKalman.
 */

#ifndef OPENKALMAN_GLOBAL_DEFINITIONS_HPP
#define OPENKALMAN_GLOBAL_DEFINITIONS_HPP

#include <type_traits>
#include <limits>
#include <functional>

namespace OpenKalman
{

  /**
   * \brief A constant indicating that a size or index is dynamic.
   * \details A dynamic size or index can be set, or change, during runtime and is not known at compile time.
   */
#ifdef __cpp_lib_span
  inline constexpr std::size_t dynamic_size = std::dynamic_extent;
#else
  inline constexpr std::size_t dynamic_size = std::numeric_limits<std::size_t>::max();
#endif


  /**
   * \brief A constant indicating that a difference in sizes or indices is dynamic.
   * \details A dynamic difference can be set, or change, during runtime and is not known at compile time.
   */
  inline constexpr std::ptrdiff_t dynamic_difference = std::numeric_limits<std::ptrdiff_t>::max();


  /**
   * \brief The layout format of a multidimensional array.
   */
  enum struct Layout : int {
    none, ///< No storage layout (e.g., if the elements are calculated rather than stored).
    right, ///< Row-major storage (C or C++ style): contiguous storage in which the right-most index has a stride of 1.
    left, ///< Column-major storage (Fortran, Matlab, or Eigen style): contiguous storage in which the left-most extent has a stride of 1.
    stride, ///< A generalization of the above: a custom stride is specified for each index.
  };


  /**
   * \brief The type of a triangular matrix.
   * \details This is generally applicable to a rank-2 tensor (e.g., a matrix).
   * It also applies to tensors of rank > 2, in which case every rank-2 slice over dimensions 0 and 1 must be a triangle of this type.
   */
  enum struct TriangleType : int {
    diagonal, ///< A diagonal matrix (both a lower-left and an upper-right triangular matrix).
    lower, ///< A lower-left triangular matrix.
    upper, ///< An upper-right triangular matrix.
    any, ///< Lower, upper, or diagonal matrix.
    // \todo strict_diagonal,
      //< A specific diagonal object for rank 2k (k:ℕ) tensors in which each element is zero
      //< unless its indices can be divided into two identical sequences.
      //< (Examples, component x[1,0,2,1,0,2] in rank-6 tensor x or component y[2,5,2,5] in rank-4 tensor y.)
  };


  /**
   * \brief The type of a hermitian adapter, indicating which triangle of the nested matrix is used.
   * \details This type can be statically cast from \ref TriangleType so that <code>lower</code>, <code>upper</code>,
   * and <code>any</code> correspond to each other. The value <code>none</code> corresponds to TriangleType::diagonal.
   *
   */
  enum struct HermitianAdapterType : int {
    any = static_cast<int>(TriangleType::diagonal), ///< Either lower or upper hermitian adapter.
    lower = static_cast<int>(TriangleType::lower), ///< A lower-left hermitian adapter.
    upper = static_cast<int>(TriangleType::upper), ///< An upper-right hermitian adapter.
  };


  /**
   * \brief How a concept or trait is qualified.
   * \details Qualification::depends_on_dynamic_shape means that the concept or trait may vary based on the
   * dynamic shape of the argument, which is not known at compile time. For example:
   * - <code>square_shaped<T, Qualification::unqualified></code> means that T is known at compile time to be a square matrix.
   * - <code>square_shaped<T, Qualification::depends_on_dynamic_shape></code> means that T <em>could</em> be a square matrix, but whether it
   * actually <em>is</em> cannot be determined at compile time.
   */
  enum struct Qualification : int {
    unqualified, ///< At compile time, the property is known to apply.
    depends_on_dynamic_shape, ///< The property is not ruled out and depends on the dynamic shape.
  };


  constexpr Qualification operator not (Qualification x)
  {
    return x == Qualification::unqualified ? Qualification::depends_on_dynamic_shape : Qualification::unqualified;
  }


  constexpr Qualification operator and (Qualification x, Qualification y)
  {
    return x == Qualification::unqualified and y == Qualification::unqualified ? Qualification::unqualified : Qualification::depends_on_dynamic_shape;
  }


  constexpr Qualification operator or (Qualification x, Qualification y)
  {
    return x == Qualification::unqualified or y == Qualification::unqualified ? Qualification::unqualified : Qualification::depends_on_dynamic_shape;
  }


  namespace internal
  {
    // ----------------------- //
    //  is_plus, is_multiplies //
    // ----------------------- //

    template<typename T>
    struct is_plus : std::false_type {};

    template<typename T>
    struct is_plus<std::plus<T>> : std::true_type {};

    template<typename T>
    struct is_multiplies : std::false_type {};

    template<typename T>
    struct is_multiplies<std::multiplies<T>> : std::true_type {};


    // ------------------------ //
    //  remove_rvalue_reference //
    // ------------------------ //

    /**
     * \brief If T is an rvalue reference, remove the reference. Otherwise, the result is T.
     */
    template<typename T>
    struct remove_rvalue_reference
    {
      using type = std::conditional_t<std::is_rvalue_reference_v<T>, std::remove_reference_t<T>, T>;
    };


    /**
     * \brief Helper type for \ref remove_rvalue_reference.
     */
    template<typename T>
    using remove_rvalue_reference_t = typename remove_rvalue_reference<T>::type;


  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_GLOBAL_DEFINITIONS_HPP
