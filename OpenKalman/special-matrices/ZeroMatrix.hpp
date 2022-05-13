/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for Eigen3::ZeroMatrix
 */

#ifndef OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
#define OPENKALMAN_EIGEN3_ZEROMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // ZeroMatrix is declared in eigen3-forward-declarations.hpp.

#ifdef __cpp_concepts
  template<indexible PatternMatrix>
#else
  template<typename PatternMatrix>
#endif
  struct ZeroMatrix : MatrixTraits<PatternMatrix>::template MatrixBaseFrom<ZeroMatrix<PatternMatrix>>
  {

  private:

    using nested_scalar = scalar_type_of_t<PatternMatrix>;
    static constexpr auto nested_rows = row_dimension_of_v<PatternMatrix>;
    static constexpr auto nested_cols = column_dimension_of_v<PatternMatrix>;

    using MyDimensions = std::decay_t<decltype(get_all_dimensions_of(std::declval<PatternMatrix>()))>;

  public:

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of \ref index_descriptor "index_descriptors" representing the
     * number of dimensions.
     */
#ifdef __cpp_concepts
    template<index_descriptor...D> requires (sizeof...(D) == max_indices_of_v<PatternMatrix>) and
      std::constructible_from<MyDimensions, const D&...>
#else
    template<typename...D,
      std::enable_if_t<(index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<PatternMatrix> and
        std::is_constructible<MyDimensions, const D&...>::value, int> = 0>
#endif
    ZeroMatrix(const D&...d) : my_dimensions {d...} {}


    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ZeroMatrix {2, 3} constructs a 2-by-3 dynamic matrix, ZeroMatrix {3} constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ZeroMatrix {} constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... I>
    requires (sizeof...(I) == number_of_dynamic_indices_v<PatternMatrix>)
#else
    template<typename...I, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
      (sizeof...(I) == number_of_dynamic_indices<PatternMatrix>::value), int> = 0>
#endif
    ZeroMatrix(const I...i)
      : my_dimensions {OpenKalman::internal::make_dimensions_tuple<PatternMatrix>(static_cast<std::size_t>(i)...)} {}


    /**
     * \brief Copy constructor
     */
    ZeroMatrix(const ZeroMatrix&) = default;


    /**
     * \brief Move constructor
     */
    ZeroMatrix(ZeroMatrix&&) = default;

  private:

    template<typename T, std::size_t I>
    static constexpr bool arg_one_dim_matches()
    {
      constexpr auto dim1 = index_dimension_of_v<T, I>;
      constexpr auto dim2 = dimension_size_of_v<std::tuple_element_t<I, MyDimensions>>;
      return dim1 == dynamic_size or dim2 == dynamic_size or dim1 == dim2;
    };


    template<typename T, std::size_t...I>
    static constexpr bool arg_matches_impl(std::index_sequence<I...>)
    {
      return (arg_one_dim_matches<T, I>() and ...);
    };


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct zero_arg_matches : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires (arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))
    struct zero_arg_matches<T>
#else
    template<typename T>
    struct zero_arg_matches<T, std::enable_if_t<arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})>>
#endif
      : std::true_type {};

  public:

    /**
     * \internal
     * \brief Construct a ZeroMatrix from another zero_matrix.
     * \tparam M A zero_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and zero_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and zero_arg_matches<M>::value, int> = 0>
#endif
    ZeroMatrix(M&& m) : my_dimensions {get_all_dimensions_of(std::forward<M>(m))} {}

  private:

    template<typename T, std::size_t...I>
    void check_runtime_sizes_impl(const T& t, std::index_sequence<I...>)
    {
      ([](const T& t, const auto& my_dims) {
        if constexpr (index_dimension_of_v<T, I> == dynamic_size)
        {
          constexpr auto t_dim = get_dimensions_of<I>(t);
          constexpr auto dim = get_dimension_size_of(std::get<I>(my_dims));
          if (t_dim != dim)
            throw std::logic_error {"In an argument to ZeroMatrix assignment operator, "
              "the dimension of index " + std::to_string(I) + " is " + std::to_string(t_dim) +
              ", which does not match index " + std::to_string(I) + " of the ZeroMatrix, which is " +
              std::to_string(dim)};;
        }
      }(t, my_dimensions), ...);
    };


    template<typename T>
    void check_runtime_sizes(const T& t)
    {
      if constexpr (has_dynamic_dimensions<T> or has_dynamic_dimensions<PatternMatrix>)
        return runtime_sizes_match_impl(t, std::make_index_sequence<max_indices_of_v<T>>{});
    };

  public:

    /**
     * \brief Copy assignment operator
     */
    ZeroMatrix& operator=(const ZeroMatrix& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Move assignment operator
     */
    ZeroMatrix& operator=(ZeroMatrix&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible zero_matrix.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and zero_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      zero_arg_matches<M>::value, int> = 0>
#endif
    auto& operator=(M&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Element accessor.
     * \brief Element accessor.
     * \note Does not do any runtime bounds checking.
     * \param d The indices
     * \return The element corresponding to the indices (always zero).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::size_t>...D> requires (sizeof...(D) <= max_indices_of_v<PatternMatrix>)
#else
    template<typename...D, std::enable_if_t<(std::is_convertible<D, const std::size_t>::value and ...) and
      (sizeof...(D) <= max_indices_of_v<PatternMatrix>), int> = 0>
#endif
    constexpr nested_scalar
    operator()(D...d) const
    {
      return 0;
    }


    /**
     * \brief Element accessor for a first-order tensor (e.g., a row or column vector).
     * \note Does not do any runtime bounds checking.
     * \param i The index.
     * \return The element at index i (always zero).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::size_t> D> requires (tensor_order_of_v<PatternMatrix> <= 1)
#else
    template<typename D, std::enable_if_t<std::is_convertible<D, const std::size_t>::value and
      tensor_order_of<PatternMatrix>::value <= 1, int> = 0>
#endif
    constexpr nested_scalar
    operator[](D d) const
    {
      return 0;
    }


    /**
     * \return a tuple containing \ref index_descriptor "index_descriptors" for the dimensions for this ZeroMatrix
     */
    auto& get_all_dimensions() & { return my_dimensions; }
    /// \overload
    const auto& get_all_dimensions() const & { return my_dimensions; }
    /// \overload
    auto&& get_all_dimensions() && { return std::move(my_dimensions); }
    /// \overload
    const auto&& get_all_dimensions() const && { return std::move(my_dimensions); }

  private:

    MyDimensions my_dimensions;

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<zero_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<zero_matrix<Arg>, int> = 0>
#endif
  ZeroMatrix(Arg&&) -> ZeroMatrix<std::conditional_t<eigen_zero_expr<Arg> or eigen_constant_expr<Arg>,
    pattern_matrix_of_t<Arg>, std::decay_t<Arg>>>;


} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
