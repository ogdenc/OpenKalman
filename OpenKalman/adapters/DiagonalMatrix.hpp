/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for DiagonalMatrix
 */

#ifndef OPENKALMAN_DIAGONALMATRIX_HPP
#define OPENKALMAN_DIAGONALMATRIX_HPP

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<vector<0, Qualification::depends_on_dynamic_shape> NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix : OpenKalman::internal::AdapterBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(vector<NestedMatrix, 0, Qualification::depends_on_dynamic_shape>);
#endif

  private:

    using Base = OpenKalman::internal::AdapterBase<DiagonalMatrix, NestedMatrix>;

    static constexpr auto dim = index_dimension_of_v<NestedMatrix, 0>;

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_constructible_from_diagonal : std::false_type {};

    template<typename T>
    struct is_constructible_from_diagonal<T, std::void_t<decltype(NestedMatrix {diagonal_of(std::declval<T>())})>>
      : std::true_type {};
#endif

  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    constexpr DiagonalMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    constexpr DiagonalMatrix()
#endif
      : Base {} {}


    /**
     * \brief Construct from a column vector
     */
#ifdef __cpp_concepts
    template<vector<0, Qualification::depends_on_dynamic_shape> Arg> requires
      (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      maybe_same_shape_as<NestedMatrix, Arg> and (vector<Arg> or not diagonal_matrix<Arg>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<vector<Arg, 0, Qualification::depends_on_dynamic_shape> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      maybe_same_shape_as<NestedMatrix, Arg> and (vector<Arg> or not diagonal_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    constexpr explicit DiagonalMatrix(Arg&& arg) : Base {std::forward<Arg>(arg)} {}

  private:

    template<typename T, std::size_t...Is>
    static constexpr bool square_dimensions_match(std::index_sequence<Is...>)
    {
      using D0 = vector_space_descriptor_of_t<NestedMatrix, 0>;
      if constexpr (dynamic_vector_space_descriptor<D0>) return true;
      else return ((dynamic_dimension<T, Is> or equivalent_to<D0, vector_space_descriptor_of_t<T, Is>>) and ...);
    }

  public:

    /**
     * \brief construct from a \ref square_shaped
     * \tparam Arg
     */
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (not vector<Arg, 0, Qualification::depends_on_dynamic_shape> or (not vector<Arg> and diagonal_matrix<Arg>)) and
      (square_dimensions_match<Arg>(std::make_index_sequence<index_count_v<Arg>>{})) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (not vector<Arg, 0, Qualification::depends_on_dynamic_shape> or (not vector<Arg> and diagonal_matrix<Arg>)) and
      (square_dimensions_match<Arg>(std::make_index_sequence<index_count_v<Arg>>{})), int> = 0>
#endif
    constexpr explicit DiagonalMatrix(Arg&& arg) : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients that define the diagonal.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
      requires(Args ... args) {
        NestedMatrix {make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
        std::is_constructible_v<NestedMatrix, untyped_dense_writable_matrix_t<NestedMatrix, Layout::none, Scalar, sizeof...(Args), 1>>, int> = 0>
#endif
    constexpr DiagonalMatrix(Args...args) : Base {make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dim * dim) and (dim > 1) and
      requires(Args ... args) {
        NestedMatrix {diagonal_of(make_dense_object_from<NestedMatrix, dim, dim, Scalar>(
          static_cast<const Scalar>(args)...))};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (sizeof...(Args) == dim * dim) and (dim > 1) and
        std::is_constructible_v<NestedMatrix, decltype(diagonal_of(
          make_dense_object_from<NestedMatrix, dim, dim, Scalar>(
            static_cast<const Scalar>(std::declval<Args>())...)))>, int> = 0>
#endif
    constexpr DiagonalMatrix(Args ... args) : Base {diagonal_of(
      make_dense_object_from<NestedMatrix, dim, dim, Scalar>(static_cast<const Scalar>(args)...))} {}

#ifndef __cpp_concepts
  private:

    template<typename Arg, typename = void>
    struct constants_match : std::true_type {};

    template<typename Arg>
    struct constants_match<Arg, std::enable_if_t<
      constant_coefficient<NestedMatrix>::value != constant_diagonal_coefficient<Arg>::value>>
        : std::false_type {};

  public:
#endif

    /// Assign from another \ref diagonal_matrix.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      maybe_same_shape_as<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))> and
      (not constant_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and
      (not requires { requires constant_coefficient<NestedMatrix>::value != constant_diagonal_coefficient<Arg>::value; }) and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of<DiagonalMatrix, std::decay_t<Arg>>::value) and
      maybe_same_shape_as<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))> and
      (not constant_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and constants_match<Arg>::value and
      std::is_assignable<std::add_lvalue_reference_t<NestedMatrix>, decltype(diagonal_of(std::declval<Arg>()))>::value, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      using Arg_diag = decltype(diagonal_of(std::declval<Arg>()));

      if constexpr (not same_shape_as<NestedMatrix, Arg_diag>)
        if (not same_shape(this->nested_object(), diagonal_of(std::declval<Arg>())))
          throw std::invalid_argument {"Argument to DiagonalMatrix assignment operator has non-matching vector space descriptors."};

      if constexpr (constant_matrix<NestedMatrix>)
      {
        if constexpr (not constant_matrix<NestedMatrix, ConstantType::static_constant> or not constant_diagonal_matrix<Arg, ConstantType::static_constant>)
          if (get_scalar_constant_value(this->nested_object()) != get_scalar_constant_value(diagonal_of(std::forward<Arg>(arg))))
            throw std::invalid_argument {"Argument to constant_diagonal DiagonalMatrix assignment operator has non-matching constant value."};
      }
      else
      {
        this->nested_object() = diagonal_of(std::forward<Arg>(arg));
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr (dynamic_dimension<NestedMatrix, 0>)
        assert(get_vector_space_descriptor<0>(this->nested_object()) == get_vector_space_descriptor<0>(arg));

      this->nested_object() += diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator-=(Arg&& arg)
    {
      if constexpr (dynamic_dimension<NestedMatrix, 0>)
        assert(get_vector_space_descriptor<0>(this->nested_object()) == get_vector_space_descriptor<0>(arg));

      this->nested_object() -= diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_object() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_object() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<(index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(index_dimension_of_v<Arg, 0> == dim);
      this->nested_object() = this->nested_object().array() * arg.nested_object().array();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return to_diagonal(-nested_object(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return to_diagonal(-nested_object(*this));
    }

    decltype(auto) operator-() const&&
    {
      return to_diagonal(-nested_object(std::move(*this)));
    }
#endif


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return to_diagonal(scalar_product(nested_object(std::forward<Arg>(arg)), s));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return to_diagonal(scalar_product(nested_object(std::forward<Arg>(arg)), s));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return to_diagonal(scalar_quotient(nested_object(std::forward<Arg>(arg)), s));
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /**
   * \brief Deduce DiagonalMatrix template parameters for a column vector.
   * \tparam Arg A column vector
   */
#ifdef __cpp_concepts
  template<vector<0, Qualification::depends_on_dynamic_shape> Arg> requires vector<Arg> or (not diagonal_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<vector<Arg, 0, Qualification::depends_on_dynamic_shape> and
    (vector<Arg> or not diagonal_matrix<Arg>), int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<Arg>>;


  /**
   * \brief Deduce DiagonalMatrix template parameters for a square matrix.
   * \tparam Arg A \ref square_shaped
   */
#ifdef __cpp_concepts
  template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires
    (index_count_v<Arg> == dynamic_size or index_count_v<Arg> <= 2) and
    (not vector<Arg, 0, Qualification::depends_on_dynamic_shape> or (not vector<Arg> and diagonal_matrix<Arg>))
#else
  template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
    (index_count<Arg>::value == dynamic_size or index_count<Arg>::value <= 2) and
    (not vector<Arg, 0, Qualification::depends_on_dynamic_shape> or (not vector<Arg> and diagonal_matrix<Arg>)), int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<decltype(diagonal_of(std::declval<Arg&&>()))>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename ColumnVector>
    struct indexible_object_traits<DiagonalMatrix<ColumnVector>>
    {
      using scalar_type = scalar_type_of_t<ColumnVector>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N)
      {
        return OpenKalman::get_vector_space_descriptor<0>(std::forward<Arg>(arg).nested_object());
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }

      // get_constant(const Arg& arg) not defined

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_coefficient {arg.nested_object()};
      }


      template<Qualification b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<ColumnVector, b>;


      template<Qualification b>
      static constexpr bool is_square = true;


      template<TriangleType t>
      static constexpr bool is_triangular = true;


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires OpenKalman::one_dimensional<nested_object_of_t<Arg&>> and directly_accessible<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<one_dimensional<typename nested_object_of<Arg&>::type> and
        directly_accessible<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(nested_object(arg)); }


      static constexpr Layout layout = OpenKalman::one_dimensional<ColumnVector> ? layout_of_v<ColumnVector> : Layout::none;

    };

  } // namespace interface

} // OpenKalman



#endif //OPENKALMAN_DIAGONALMATRIX_HPP
