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
 * \brief Overloaded general conversion functions.
 */

#ifndef OPENKALMAN_CONVERSION_FUNCTIONS_HPP
#define OPENKALMAN_CONVERSION_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires column_vector<Arg> or dynamic_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<column_vector<Arg> or dynamic_columns<Arg>, int> = 0>
#endif
  inline decltype(auto)
  to_diagonal(Arg&& arg)
  {
    constexpr auto dim = row_dimension_of_v<Arg>;

    if constexpr (dim == 1)
    {
      if constexpr (dynamic_columns<Arg>)
        if (get_index_dimension_of<1>(arg) != 1) throw std::domain_error {
          "Argument of to_diagonal must be a column vector, not a row vector"};
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> and dim != dynamic_size)
    {
      // note, the interface function should deal with a zero matrix of uncertain size.

      if constexpr (dynamic_columns<Arg>)
        if (get_index_dimension_of<1>(arg) != 1) throw std::domain_error {
          "Argument of to_diagonal must have 1 column; instead it has " +
          std::to_string(get_index_dimension_of<1>(arg))};
      return make_zero_matrix_like<Arg>(Dimensions<dim>{}, Dimensions<dim>{});
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
  }


  namespace detail
  {
    template<typename Arg>
    inline void check_if_square_at_runtime(const Arg& arg)
    {
      if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
        throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead, " +
        (get_index_dimension_of<0>(arg) == get_index_dimension_of<1>(arg) ?
          "the row and column indices have non-equivalent types" :
          "it has " + std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + "columns")};
    };
  }


  // ================================== //
  //  Modular transformation functions  //
  // ================================== //

#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_euclidean(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In to_euclidean, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In to_euclidean, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      return interface::ModularTransformationTraits<Arg>::to_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    return to_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_euclidean(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In from_euclidean, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In from_euclidean, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      return interface::ModularTransformationTraits<Arg>::from_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg)
  {
    return from_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg, const C& c)
  {
    if constexpr (euclidean_index_descriptor<C> or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_euclidean(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In wrap_angles, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In wrap_angles, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      interface::ModularTransformationTraits<Arg>::wrap_angles(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    return wrap_angles(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
    // ------------------------ //
    //  to_covariance_nestable  //
    // ------------------------ //

    /**
     * \overload
     * \internal
     * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \internal
     * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \tparam Arg A \ref covariance or \ref typed_matrix.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return The result of converting Arg to a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
        (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_CONVERSION_FUNCTIONS_HPP
