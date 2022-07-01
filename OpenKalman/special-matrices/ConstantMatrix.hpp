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
 * \brief Definitions for Eigen3::ConstantMatrix
 */

#ifndef OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
#define OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // ConstantMatrix is declared in forward-class-declarations.hpp.

#ifdef __cpp_concepts
# if __cpp_nontype_template_args >= 201911L
  template<indexible PatternMatrix, scalar_type_of_t<PatternMatrix> constant>
# else
  template<indexible PatternMatrix, auto constant> requires
    std::convertible_to<decltype(constant), scalar_type_of_t<PatternMatrix>>
# endif
#else
  template<typename PatternMatrix, auto constant>
#endif
  struct ConstantMatrix : MatrixTraits<PatternMatrix>::template MatrixBaseFrom<ConstantMatrix<PatternMatrix, constant>>
  {

  private:

    using nested_scalar = scalar_type_of_t<PatternMatrix>;
    static constexpr std::size_t max_indices = max_indices_of_v<PatternMatrix>;

#ifndef __cpp_concepts
    static_assert(std::is_convertible_v<decltype(constant), nested_scalar>);
#endif


    using MyDimensions = decltype(get_all_dimensions_of(std::declval<PatternMatrix>()));

  public:

    /**
     * \brief Construct a ConstantMatrix.
     * \details The constructor can take a number of \ref index_descriptor "index_descriptors" representing the
     * number of dimensions.
     */
#ifdef __cpp_concepts
    template<euclidean_index_descriptor...D> requires (sizeof...(D) == max_indices) and
      std::constructible_from<MyDimensions, const D&...>
#else
    template<typename...D,
      std::enable_if_t<(euclidean_index_descriptor<D> and ...) and sizeof...(D) == max_indices and
        std::is_constructible<MyDimensions, const D&...>::value, int> = 0>
#endif
    constexpr ConstantMatrix(const D&...d) : my_dimensions {d...} {}

  private:

    template<std::size_t I>
    static constexpr auto make_dimensions_tuple()
    {
      using Dim = std::tuple_element_t<I, MyDimensions>;
      if constexpr (I >= max_indices)
        return std::tuple{};
      else
        return std::tuple_cat(std::tuple {Dim {Dimensions<index_dimension_of_v<PatternMatrix, I>>{}}},
          make_dimensions_tuple<I + 1>());
    }


    template<std::size_t I, typename N, typename...Ns>
    static constexpr auto make_dimensions_tuple(N n, Ns...ns)
    {
      using Dim = std::tuple_element_t<I, MyDimensions>;
      if constexpr (dynamic_dimension<PatternMatrix, I>)
        return std::tuple_cat(std::tuple {Dim {Dimensions{n}}}, make_dimensions_tuple<I + 1>(ns...));
      else
        return std::tuple_cat(std::tuple {Dim {Dimensions<index_dimension_of_v<PatternMatrix, I>>{}}},
          make_dimensions_tuple<I + 1>(n, ns...));
    }

  public:

    /**
     * \brief Construct a ConstantMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, ConstantMatrix(2, 3) constructs a 2-by-3 dynamic matrix, ConstantMatrix(3) constructs a
     * 2-by-3 matrix in which there are two fixed row dimensions and three dynamic column dimensions, and
     * ConstantMatrix() constructs a fixed matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... I> requires
      (sizeof...(I) == number_of_dynamic_indices_v<PatternMatrix>) and
      requires(const I...i) { MyDimensions {make_dimensions_tuple<0>(static_cast<const std::size_t>(i)...)}; }
#else
    template<typename...I, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
      (sizeof...(I) == number_of_dynamic_indices<PatternMatrix>::value) and
      std::is_constructible<MyDimensions, decltype(make_dimensions_tuple<0>(static_cast<const std::size_t>(std::declval<I>())...))>::value, int> = 0>
#endif
    constexpr ConstantMatrix(const I...i)
      : my_dimensions {make_dimensions_tuple<0>(static_cast<const std::size_t>(i)...)} {}


    /**
     * \brief Copy constructor
     */
    constexpr ConstantMatrix(const ConstantMatrix&) = default;


    /**
     * \brief Move constructor
     */
    constexpr ConstantMatrix(ConstantMatrix&&) = default;

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
    struct constant_arg_matches : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires (constant_coefficient_v<std::decay_t<T>> == constant) and
      (arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))
    struct constant_arg_matches<T>
#else
    template<typename T>
    struct constant_arg_matches<T, std::enable_if_t<(constant_coefficient<std::decay_t<T>>::value == constant) and
      arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})>>
#endif
      : std::true_type {};

  public:

    /**
     * \internal
     * \brief Construct a ConstantMatrix from another constant matrix.
     * \tparam M A constant_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<constant_matrix M> requires (not std::same_as<M, ConstantMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and constant_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and (not std::is_same_v<M, ConstantMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and constant_arg_matches<M>::value, int> = 0>
#endif
    constexpr ConstantMatrix(M&& m) : my_dimensions {get_all_dimensions_of(std::forward<M>(m))} {}

  private:

    template<typename T, std::size_t...I>
    constexpr void check_runtime_sizes_impl(const T& t, std::index_sequence<I...>)
    {
      ([](const T& t, const auto& my_dims) {
        if constexpr (index_dimension_of_v<T, I> == dynamic_size)
        {
          constexpr auto t_dim = get_dimensions_of<I>(t);
          constexpr auto dim = get_dimension_size_of(std::get<I>(my_dims));
          if (t_dim != dim)
            throw std::logic_error {"In an argument to ConstantMatrix assignment operator, "
              "the dimension of index " + std::to_string(I) + " is " + std::to_string(t_dim) +
              ", which does not match index " + std::to_string(I) + " of the ConstantMatrix, which is " +
              std::to_string(dim)};;
        }
      }(t, my_dimensions), ...);
    };


    template<typename T>
    constexpr void check_runtime_sizes(const T& t)
    {
      if constexpr (has_dynamic_dimensions<T> or has_dynamic_dimensions<PatternMatrix>)
        return runtime_sizes_match_impl(t, std::make_index_sequence<max_indices_of_v<T>>{});
    };

  public:

    /**
     * \brief Copy assignment operator
     */
    constexpr ConstantMatrix& operator=(const ConstantMatrix& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Move assignment operator
     */
    constexpr ConstantMatrix& operator=(ConstantMatrix&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix M> requires (not std::same_as<M, ConstantMatrix>) and constant_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<constant_matrix<M> and (not std::is_same_v<M, ConstantMatrix>) and
      constant_arg_matches<M>::value, int> = 0>
#endif
    constexpr auto& operator=(M&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Element accessor.
     * \note Does not do any runtime bounds checking.
     * \param d The indices
     * \return The element corresponding to the indices (always the constant).
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
      return constant;
    }


    /**
     * \brief Element accessor for a first-order tensor (e.g., a row or column vector).
     * \note Does not do any runtime bounds checking.
     * \param i The index.
     * \return The element at index i (always the constant).
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
      return constant;
    }

  private:

    MyDimensions my_dimensions;

#ifdef __cpp_concepts
    template<typename T, std::size_t N> friend struct interface::IndexTraits;
    template<typename T, std::size_t N> friend struct interface::CoordinateSystemTraits;
#else
    template<typename T, std::size_t N, typename Enable> friend struct interface::IndexTraits;
    template<typename T, std::size_t N, typename Enable> friend struct interface::CoordinateSystemTraits;
#endif

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(ConstantMatrix<NestedMatrix, constant>&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(const ConstantMatrix<NestedMatrix, constant>&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(ConstantMatrix<NestedMatrix, constant>&&) -> ConstantMatrix<NestedMatrix, constant>;

  template<typename NestedMatrix, auto constant>
  ConstantMatrix(const ConstantMatrix<NestedMatrix, constant>&&) -> ConstantMatrix<NestedMatrix, constant>;


  #ifdef __cpp_concepts
#  if __cpp_nontype_template_args >= 201911L
  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>)
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, constant_coefficient_v<Arg>>;
#  else
  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>) and std::is_integral_v<scalar_type_of_t<Arg>>
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, constant_coefficient_v<Arg>>;

  template<constant_matrix Arg> requires (not eigen_constant_expr<Arg>) and
    (not std::is_integral_v<scalar_type_of_t<Arg>>) and
    (constant_coefficient_v<Arg> == static_cast<std::intmax_t>(constant_coefficient_v<Arg>))
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#  endif
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and not eigen_constant_expr<Arg> and
    constant_coefficient_v<Arg> == static_cast<std::intmax_t>(constant_coefficient_v<Arg>), int> = 0>
  ConstantMatrix(Arg&&) -> ConstantMatrix<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#endif

} // namespace OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_CONSTANTMATRIX_HPP
