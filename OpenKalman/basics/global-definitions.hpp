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
   * \brief The applicability of a concept, trait, or restraint.
   * \details Determines whether something is necessarily applicable, or alternatively just permissible, at compile time.
   * Examples:
   * - <code>square_shaped<T, Applicability::guaranteed></code> means that T is known at compile time to be square shaped.
   * - <code>square_shaped<T, Applicability::permitted></code> means that T <em>could</em> be square shaped,
   * but whether it actually <em>is</em> cannot be determined at compile time.
   */
  enum struct Applicability : int {
    guaranteed, ///< The concept, trait, or restraint represents a compile-time guarantee.
    permitted, ///< The concept, trait, or restraint is permitted, but whether it applies is not necessarily known at compile time.
  };


  constexpr Applicability operator not (Applicability x)
  {
    return x == Applicability::guaranteed ? Applicability::permitted : Applicability::guaranteed;
  }


  constexpr Applicability operator and (Applicability x, Applicability y)
  {
    return x == Applicability::guaranteed and y == Applicability::guaranteed ? Applicability::guaranteed : Applicability::permitted;
  }


  constexpr Applicability operator or (Applicability x, Applicability y)
  {
    return x == Applicability::guaranteed or y == Applicability::guaranteed ? Applicability::guaranteed : Applicability::permitted;
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


    // -------------------- //
    //  is_initializer_list //
    // -------------------- //

    /**
     * \brief Whether the argument is a specialization of std::initializer_list
     */
    template<typename T>
    struct is_initializer_list : std::false_type {};

    /// \overload
    template<typename T>
    struct is_initializer_list<std::initializer_list<T>> : std::true_type {};

    /// \overload
    template<typename T>
    struct is_initializer_list<T&> : is_initializer_list<T> {};

    /// \overload
    template<typename T>
    struct is_initializer_list<T&&> : is_initializer_list<T> {};

    /// \overload
    template<typename T>
    struct is_initializer_list<const T> : is_initializer_list<T> {};

    /// \overload
    template<typename T>
    struct is_initializer_list<volatile T> : is_initializer_list<T> {};


  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_GLOBAL_DEFINITIONS_HPP
