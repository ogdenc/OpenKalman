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
 * \brief Definitions for TriangularMatrix
 */

#ifndef OPENKALMAN_TRIANGULARMATRIX_HPP
#define OPENKALMAN_TRIANGULARMATRIX_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, TriangleType triangle_type>
    requires (max_indices_of_v<NestedMatrix> <= 2)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(square_matrix<NestedMatrix, Likelihood::maybe>);
    static_assert(max_indices_of_v<NestedMatrix> <= 2);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;

    static constexpr auto dim = dynamic_dimension<NestedMatrix, 0> ? index_dimension_of_v<NestedMatrix, 1> :
      index_dimension_of_v<NestedMatrix, 0>;

    template<typename Arg>
    static bool constexpr dimensions_match = dimension_size_of_index_is<Arg, 0, dim, Likelihood::maybe> and
      dimension_size_of_index_is<Arg, 1, dim, Likelihood::maybe>;

  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    TriangularMatrix()
#endif
      : Base {} {}


    /// Construct from a triangular adapter if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<triangular_adapter Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      triangular_matrix<Arg, triangle_type> and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<nested_matrix_of_t<Arg>> and
# if OPENKALMAN_CPP_FEATURE_CONCEPTS
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } //-- not accepted in GCC 10.1.0
# else
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
# endif
#else
    template<typename Arg, std::enable_if_t<triangular_adapter<Arg> and (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<typename nested_matrix_of<Arg>::type> and
      std::is_constructible<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular or square matrix if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not triangular_matrix<Arg, triangle_type>) and
      (not diagonal_matrix<NestedMatrix>) and dimensions_match<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      (not triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg)) throw std::invalid_argument {
            "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
            std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
            " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))
    } {}


    /// Construct from a triangular, non-adapter matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires (not triangular_adapter<Arg>) and
      (not has_nested_matrix<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and (not triangular_adapter<Arg>) and
      (not has_nested_matrix<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::is_constructible<NestedMatrix, Arg&&>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      diagonal_matrix<NestedMatrix> and dimensions_match<Arg> and (not std::constructible_from<NestedMatrix, Arg&&>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<NestedMatrix> and dimensions_match<Arg> and (not std::is_constructible_v<NestedMatrix, Arg&&>) and
      std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular square matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      dimensions_match<Arg> and requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      dimensions_match<Arg> and std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg)) throw std::invalid_argument {
            "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
            std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
            " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))
    } {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if triangle_type is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args...args) { NestedMatrix {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args)
      : Base {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if NestedMatrix is not \ref eigen_diagonal_expr but triangle_type is TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args)
      : Base {MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)} {}


    /// Assign from another \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires
      (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix, CompileTimeStatus::known> or
        requires { requires constant_diagonal_coefficient<NestedMatrix>::value == constant_diagonal_coefficient<Arg>::value; }) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not constant_diagonal_matrix<NestedMatrix>)
        set_triangle<triangle_type>(this->nested_matrix(), std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() + std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() - std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() * s);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() / s);
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg>
#else
    template<typename Arg, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      contract_in_place(this->nested_matrix(), make_dense_writable_matrix_from(arg));
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<triangular_matrix<TriangleType::any, Likelihood::maybe> M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M, TriangleType::any, Likelihood::maybe>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<
    std::conditional_t<triangular_adapter<M>, passable_t<nested_matrix_of_t<M&&>>, passable_t<M>>,
    triangle_type_of_v<M>>;


#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> M> requires (not triangular_matrix<M, TriangleType::any, Likelihood::maybe>)
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M, Likelihood::maybe> and
    (not triangular_matrix<M, TriangleType::any, Likelihood::maybe>), int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<
    std::conditional_t<hermitian_adapter<M>, passable_t<nested_matrix_of_t<M&&>>, passable_t<M>>,
    hermitian_adapter<M, HermitianAdapterType::upper> ? TriangleType::upper : TriangleType::lower>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M> requires (not triangular_matrix<M>)
#else
  template<
    TriangleType t = TriangleType::lower, typename M, std::enable_if_t<not triangular_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<TriangleType t, triangular_matrix<t> M>
#else
  template<TriangleType t, typename M, std::enable_if_t<triangular_matrix<M, t>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    if constexpr (eigen_triangular_expr<M>)
      return make_EigenTriangularMatrix<t>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenTriangularMatrix<t>(std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<triangular_matrix M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    if constexpr (eigen_triangular_expr<M>)
      return make_EigenTriangularMatrix<triangle_type_of_v<M>>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenTriangularMatrix<triangle_type_of_v<M>>(std::forward<M>(m));
  }


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Nested, TriangleType t>
    struct IndexTraits<TriangularMatrix<Nested, t>>
    {
      static constexpr std::size_t max_indices = 2;

      template<std::size_t N>
      static constexpr std::size_t dimension = dynamic_dimension<Nested, 0> ?
        index_dimension_of_v<Nested, 1> : index_dimension_of_v<Nested, 0>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_dimension<Nested, 0>)
        {
          if constexpr (dynamic_dimension<Nested, 1>)
            return get_index_dimension_of<0>(nested_matrix(arg));
          else
            return index_dimension_of_v<Nested, 1>;
        }
        else
        {
          return dimension<N>;
        }
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<Nested, b>;
    };


    template<typename NestedMatrix, TriangleType triangle_type>
    struct Elements<TriangularMatrix<NestedMatrix, triangle_type>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;


  #ifdef __cpp_lib_concepts
      template<diagonal_matrix Arg, typename I> requires element_gettable<nested_matrix_of_t<Arg&&>, 1> or
        element_gettable<nested_matrix_of_t<Arg&&>, 2>
  #else
      template<typename Arg, typename I, std::enable_if_t<diagonal_matrix<Arg> and
        element_gettable<typename nested_matrix_of<Arg&&>::type, 1> and
        element_gettable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
  #endif
      static constexpr auto get(Arg&& arg, I i)
      {
        if constexpr (element_gettable<nested_matrix_of_t<Arg&&>, 1>)
          return get_element(nested_matrix(std::forward<Arg>(arg)), i);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_gettable<nested_matrix_of_t<Arg&&>, 2>
  #else
      template<typename Arg, typename I, typename J, std::enable_if_t<
        element_gettable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
  #endif
      static constexpr scalar_type_of_t<Arg> get(Arg&& arg, I i, J j)
      {
        if (triangular_matrix<Arg, TriangleType::lower> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
        else
          return 0;
      }


  #ifdef __cpp_lib_concepts
      template<diagonal_matrix Arg, typename I> requires element_settable<nested_matrix_of_t<Arg&&>, 1> or
        element_settable<nested_matrix_of_t<Arg&&>, 2>
  #else
      template<typename Arg, typename I, std::enable_if_t<diagonal_matrix<Arg> and
        element_settable<typename nested_matrix_of<Arg&&>::type, 1> and
        element_settable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
  #endif
      static Arg&& set(Arg&& arg, const scalar_type_of_t<Arg>& s, I i)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg&&>, 1>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, static_cast<I>(1));
        return std::forward<Arg>(arg);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_settable<nested_matrix_of_t<Arg&&>, 2>
  #else
      template<typename Arg, typename I, typename J, std::enable_if_t<
        element_settable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
  #endif
      static Arg&& set(Arg&& arg, const scalar_type_of_t<Arg>& s, I i, J j)
      {
        if (triangular_matrix<Arg, TriangleType::lower> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
          set_element(nested_matrix(arg), s, i, j);
        else if (s != 0)
          throw std::out_of_range("Cannot set elements of a triangular matrix to non-zero values outside the triangle.");
        return std::forward<Arg>(arg);
      }
    };

  }

  template<typename NestedMatrix, TriangleType triangle_type>
  struct Dependencies<TriangularMatrix<NestedMatrix, triangle_type>>
  {
    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<NestedMatrix>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nested_matrix();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      auto n = make_self_contained(get_nested_matrix<0>(std::forward<Arg>(arg)));
      return TriangularMatrix<decltype(n), triangle_type> {std::move(n)};
    }
  };


} // namespace OpenKalman


#endif //OPENKALMAN_TRIANGULARMATRIX_HPP

