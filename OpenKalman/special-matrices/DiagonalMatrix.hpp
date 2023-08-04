/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
  template<vector<0, Likelihood::maybe> NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix : OpenKalman::internal::MatrixBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(indexible<NestedMatrix>);
    static_assert(vector<NestedMatrix, 0, Likelihood::maybe>);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, NestedMatrix>;

    static constexpr auto dim = row_dimension_of_v<NestedMatrix>;

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
    template<vector<0, Likelihood::maybe> Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or equivalent_to<index_descriptor_of_t<Arg, 0>, index_descriptor_of_t<NestedMatrix, 0>>) and
      std::is_constructible_v<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<vector<Arg, 0, Likelihood::maybe> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedMatrix, 0> or equivalent_to<index_descriptor_of_t<Arg, 0>, index_descriptor_of_t<NestedMatrix, 0>>), int> = 0>
#endif
    constexpr explicit DiagonalMatrix(Arg&& arg) : Base {std::forward<Arg>(arg)} {}

  private:

    template<typename T, std::size_t...Is>
    static constexpr bool square_dimensions_match(std::index_sequence<Is...>)
    {
      using D0 = index_descriptor_of_t<NestedMatrix, 0>;
      if constexpr (dynamic_index_descriptor<D0>) return true;
      else return ((dynamic_dimension<T, Is> or equivalent_to<D0, index_descriptor_of_t<T, Is>>) and ...);
    }

  public:

    /**
     * \brief construct from a \ref square_matrix
     * \tparam Arg
     */
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (not vector<Arg, 0, Likelihood::maybe>) and (square_dimensions_match<Arg>(std::make_index_sequence<max_indices_of_v<Arg>>{})) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and (not vector<Arg, 0, Likelihood::maybe>) and
      (square_dimensions_match<Arg>(std::make_index_sequence<max_indices_of_v<Arg>>{})), int> = 0>
#endif
    constexpr explicit DiagonalMatrix(Arg&& arg) : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients that define the diagonal.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
      requires(Args ... args) {
        NestedMatrix {make_dense_writable_matrix_from<NestedMatrix>(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
        std::is_constructible_v<NestedMatrix, untyped_dense_writable_matrix_t<NestedMatrix, Scalar, sizeof...(Args), 1>>, int> = 0>
#endif
    constexpr DiagonalMatrix(Args...args) : Base {make_dense_writable_matrix_from<NestedMatrix>(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dim * dim) and (dim > 1) and
      requires(Args ... args) {
        NestedMatrix {diagonal_of(make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(
          static_cast<const Scalar>(args)...))};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (sizeof...(Args) == dim * dim) and (dim > 1) and
        std::is_constructible_v<NestedMatrix, decltype(diagonal_of(
          make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(
            static_cast<const Scalar>(std::declval<Args>())...)))>, int> = 0>
#endif
    constexpr DiagonalMatrix(Args ... args) : Base {diagonal_of(
      make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(static_cast<const Scalar>(args)...))} {}


    /// Assign from another \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (row_dimension_of_v<Arg> == dim) and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (row_dimension_of<Arg>::value == dim) and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(other));
      }
      return *this;
    }


    /// Assign from a \ref zero_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<zero_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not identity_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and square_matrix<Arg, Likelihood::maybe>
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and square_matrix<Arg, Likelihood::maybe>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix>)
      {
        if constexpr (has_dynamic_dimensions<Arg>) assert(get_index_descriptor<0>(arg) == get_index_descriptor<1>(arg));
        if constexpr (dynamic_rows<NestedMatrix>) assert(get_index_descriptor<0>(this->nested_matrix()) == get_index_descriptor<0>(arg));

        this->nested_matrix() = make_zero_matrix_like(this->nested_matrix());
      }
      return *this;
    }


    /// Assign from an \ref identity_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<identity_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and square_matrix<Arg, Likelihood::maybe>
#else
    template<typename Arg, std::enable_if_t<
      identity_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and square_matrix<Arg, Likelihood::maybe>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      if constexpr (not identity_matrix<NestedMatrix>)
      {
        if constexpr (has_dynamic_dimensions<Arg>) assert(get_index_descriptor<0>(arg) == get_index_descriptor<1>(arg));
        if constexpr (dynamic_rows<NestedMatrix>) assert(get_index_descriptor<0>(this->nested_matrix()) == get_index_descriptor<0>(arg));

        this->nested_matrix() = make_constant_matrix_like<NestedMatrix, scalar_type_of_t<NestedMatrix>, 1>(Dimensions<dim>{}, Dimensions<1>{});
      }
      return *this;
    }


    /// Assign from a general diagonal matrix, other than \ref eigen_diagonal_expr zero_matrix, or identity_matrix.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_index_descriptor<0>(this->nested_matrix()) == get_index_descriptor<0>(arg));

      this->nested_matrix() = diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_index_descriptor<0>(this->nested_matrix()) == get_index_descriptor<0>(arg));

      this->nested_matrix() += diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator-=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_index_descriptor<0>(this->nested_matrix()) == get_index_descriptor<0>(arg));

      this->nested_matrix() -= diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<(row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(row_dimension_of_v<Arg> == dim);
      this->nested_matrix() = this->nested_matrix().array() * arg.nested_matrix().array();
      return *this;
    }


    auto square() const
    {
      auto b = this->nested_matrix().array().square().matrix();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


    auto square_root() const
    {
      auto b = this->nested_matrix().cwiseSqrt();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return to_diagonal(-nested_matrix(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return to_diagonal(-nested_matrix(*this));
    }

    decltype(auto) operator-() const&&
    {
      return to_diagonal(-nested_matrix(std::move(*this)));
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
      return to_diagonal(nested_matrix(std::forward<Arg>(arg)) * s);
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return to_diagonal(s * nested_matrix(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return to_diagonal(nested_matrix(std::forward<Arg>(arg)) / s);
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#if not defined(__cpp_concepts) or not OPENKALMAN_CPP_FEATURE_CONCEPTS
  namespace detail
  {
    template<typename T, typename = void>
    struct diagonal_exists : std::false_type {};

    template<typename T>
    struct diagonal_exists<T, std::void_t<decltype(diagonal_of(std::declval<T>()))>> : std::true_type {};
  };
#endif


  /**
   * \brief Deduce DiagonalMatrix template parameters for a column vector.
   * \tparam Arg A column vector
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<vector<0, Likelihood::maybe> Arg> requires (not square_matrix<Arg>) or
    (not requires(Arg&& arg) { diagonal_of(std::forward<Arg>(arg)); })
#else
  template<typename Arg, std::enable_if_t<vector<Arg, 0, Likelihood::maybe> and
    (not square_matrix<Arg> or not detail::diagonal_exists<Arg>::value), int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<Arg>>;


  /**
   * \brief Deduce DiagonalMatrix template parameters for a square matrix.
   * \tparam Arg A \ref square_matrix
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<square_matrix<Likelihood::maybe> Arg>
  requires (square_matrix<Arg> or not vector<Arg, 0, Likelihood::maybe>) and
    requires(Arg&& arg) { diagonal_of(std::forward<Arg>(arg)); }
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
    (square_matrix<Arg> or not vector<Arg, 0, Likelihood::maybe>) and
    detail::diagonal_exists<Arg>::value, int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<decltype(diagonal_of(std::declval<Arg&&>()))>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename ColumnVector>
    struct IndexibleObjectTraits<DiagonalMatrix<ColumnVector>>
    {
      static constexpr std::size_t max_indices = 2;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        return OpenKalman::get_index_descriptor<0>(nested_matrix(arg));
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<ColumnVector, b>;

      template<Likelihood b>
      static constexpr bool is_square = true;

      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<ColumnVector>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return DiagonalMatrix {make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)))};
      }

      // get_constant(const Arg& arg) not defined

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_coefficient {arg.nested_matrix()};
      }

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      template<Likelihood b>
      static constexpr bool is_diagonal_adapter = true;


      using scalar_type = scalar_type_of_t<ColumnVector>;


#ifdef __cpp_lib_concepts
      template<typename Arg, typename I>
      requires element_gettable<nested_matrix_of_t<Arg>, 1> or element_gettable<nested_matrix_of_t<Arg&&>, 2>
#else
      template<typename Arg, typename I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg&&>::type, 1> or
        element_gettable<typename nested_matrix_of<Arg>::type, 2>, int> = 0>
#endif
      static constexpr auto get(Arg&& arg, I i)
      {
        if constexpr (element_gettable<nested_matrix_of_t<Arg&&>, 1>)
          return get_element(nested_matrix(std::forward<Arg>(arg)), i);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, static_cast<I>(1));
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_gettable<nested_matrix_of_t<Arg&&>, 2> or
        element_gettable<nested_matrix_of_t<Arg&&>, 1>
#else
      template<typename Arg, typename I, typename J, std::enable_if_t<
        element_gettable<typename nested_matrix_of<Arg&&>::type, 2> or
        element_gettable<typename nested_matrix_of<Arg&&>::type, 1>, int> = 0>
#endif
      static constexpr scalar_type_of_t<Arg> get(Arg&& arg, I i, J j)
      {
        if (i == static_cast<I>(j))
        {
          if constexpr (element_gettable<nested_matrix_of_t<Arg&&>, 1>)
            return get_element(nested_matrix(std::forward<Arg>(arg)), i);
          else
            return get_element(nested_matrix(std::forward<Arg>(arg)), i, static_cast<J>(1));
        }
        else
        {
          return 0;
        }
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename I> requires element_settable<nested_matrix_of_t<Arg&>, 1> or
        element_settable<nested_matrix_of_t<Arg&>, 2>
#else
      template<typename Arg, typename I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg&>::type, 1> or
        element_gettable<typename nested_matrix_of<Arg&>::type, 2>, int> = 0>
#endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, static_cast<I>(1));
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_settable<nested_matrix_of_t<Arg&>, 2> or
        element_settable<nested_matrix_of_t<Arg&>, 1>
#else
      template<typename Arg, typename I, typename J, std::enable_if_t<element_settable<typename nested_matrix_of<Arg&>::type, 2> or
        element_settable<typename nested_matrix_of<Arg&>::type, 1>, int> = 0>
#endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, J j)
      {
        if (i == static_cast<I>(j))
        {
          if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
            set_element(nested_matrix(arg), s, i);
          else
            set_element(nested_matrix(arg), s, i, static_cast<I>(1));
        }
        else if (s != 0)
          throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
      }


      static constexpr bool is_writable = false;

    };

  } // namespace interface

} // OpenKalman



#endif //OPENKALMAN_DIAGONALMATRIX_HPP
