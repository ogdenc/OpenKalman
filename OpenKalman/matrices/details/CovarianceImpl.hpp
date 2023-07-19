/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for CovarianceImpl.
 */

#ifndef OPENKALMAN_COVARIANCEIMPL_HPP
#define OPENKALMAN_COVARIANCEIMPL_HPP


namespace OpenKalman::internal
{

  // ---------------- //
  //  CovarianceImpl  //
  // ---------------- //

  template<typename Derived, typename NestedMatrix>
  struct CovarianceImpl : CovarianceBase<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(covariance<Derived>);
    static_assert(covariance_nestable<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = CovarianceBase<Derived, NestedMatrix>;

  protected:
    using typename Base::CholeskyNestedMatrix;
    using Base::nested_matrix;
    using Base::cholesky_nested_matrix;
    using Base::synchronization_direction;
    using Base::synchronize_forward;
    using Base::synchronize_reverse;
    using Base::mark_nested_matrix_changed;
    using Base::mark_cholesky_nested_matrix_changed;
    using Base::mark_synchronized;


  public:

    using Base::Base;

    using Base::operator=;


    /**
     * \internal
     * \brief Make a Covariance based on an operation on the nested matrices.
     * \tparam F1 Callable operation on NestedMatrix.
     * \tparam F2 Callable operation on the return value of cholesky_nested_matrix
     */
#ifdef __cpp_concepts
    template<std::invocable<NestedMatrix&> F1, std::invocable<CholeskyNestedMatrix&> F2>
#else
    template<typename F1, typename F2, std::enable_if_t<
      std::is_invocable_v<F1, NestedMatrix&> and std::is_invocable_v<F2, CholeskyNestedMatrix&>, int> = 0>
#endif
    auto covariance_op(const F1& f1, const F2& f2) const &
    {
      using Derived2 = decltype(make_dense_writable_matrix_from<Derived>(f1(nested_matrix())));
      using R = equivalent_self_contained_t<Derived2>;
      if (synchronization_direction() >= 0)
      {
        R r {f1(nested_matrix())};
        if constexpr (not case1or2<Derived, decltype(f1(nested_matrix()))>)
        {
          if (r.synchronization_direction() > 0 and synchronization_direction() <= 0)
          {
            r.cholesky_nested_matrix() =
              to_covariance_nestable<decltype(r.cholesky_nested_matrix())>(f2(cholesky_nested_matrix()));
            r.mark_synchronized();
          }
        }
        return r;
      }
      else
      {
        return R {f2(cholesky_nested_matrix())};
      }
    }


    /// \overload \todo delete this
#ifdef __cpp_concepts
    template<std::invocable<NestedMatrix&> F1, std::invocable<CholeskyNestedMatrix&> F2>
#else
    template<typename F1, typename F2, std::enable_if_t<
      std::is_invocable_v<F1, NestedMatrix&> and std::is_invocable_v<F2, CholeskyNestedMatrix&>, int> = 0>
#endif
    auto covariance_op(const F1& f1, const F2& f2) const &&
    {
      using Derived2 = decltype(make_dense_writable_matrix_from<Derived>(f1(std::move(*this).nested_matrix())));
      using R = equivalent_self_contained_t<Derived2>;
      if (synchronization_direction() >= 0)
      {
        R r {f1(std::move(*this).nested_matrix())};
        if constexpr (not case1or2<Derived, decltype(f1(std::move(*this).nested_matrix()))>)
        {
          if (r.synchronization_direction() > 0 and synchronization_direction() <= 0)
          {
            r.cholesky_nested_matrix() =
              to_covariance_nestable<decltype(r.cholesky_nested_matrix())>(f2(std::move(*this).cholesky_nested_matrix()));
            r.mark_synchronized();
          }
        }
        return r;
      }
      else
      {
        return R {f2(std::move(*this).cholesky_nested_matrix())};
      }
    }


    /// \overload \todo delete this
#ifdef __cpp_concepts
    template<std::invocable<NestedMatrix&> F> requires std::invocable<F, CholeskyNestedMatrix&>
#else
    template<typename F, std::enable_if_t<
      std::is_invocable_v<F, NestedMatrix&> and std::is_invocable_v<F, CholeskyNestedMatrix&>, int> = 0>
#endif
    auto covariance_op(const F& f) const &
    {
      return covariance_op(f, f);
    }


    /// \overload \todo delete this
#ifdef __cpp_concepts
    template<std::invocable<NestedMatrix&> F> requires std::invocable<F, CholeskyNestedMatrix&>
#else
    template<typename F, std::enable_if_t<
      std::is_invocable_v<F, NestedMatrix&> and std::is_invocable_v<F, CholeskyNestedMatrix&>, int> = 0>
#endif
    auto covariance_op(const F& f) const &&
    {
      return std::move(*this).covariance_op(f, f);
    }

  protected:

    template<bool direct_nested>
    decltype(auto) get_nested_matrix_impl() &
    {
      if constexpr (direct_nested)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        return nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        return cholesky_nested_matrix();
      }
    }


    template<bool direct_nested>
    decltype(auto) get_nested_matrix_impl() const &
    {
      if constexpr (direct_nested)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        return nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        return cholesky_nested_matrix();
      }
    }


    template<bool direct_nested>
    decltype(auto) get_nested_matrix_impl() &&
    {
      if constexpr (direct_nested)
      {
        if (synchronization_direction() < 0) std::move(*this).synchronize_reverse();
        return std::move(*this).nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) std::move(*this).synchronize_forward();
        return std::move(*this).cholesky_nested_matrix();
      }
    }


    template<bool direct_nested>
    decltype(auto) get_nested_matrix_impl() const &&
    {
      if constexpr (direct_nested)
      {
        if (synchronization_direction() < 0) std::move(*this).synchronize_reverse();
        return std::move(*this).nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) std::move(*this).synchronize_forward();
        return std::move(*this).cholesky_nested_matrix();
      }
    }

  public:

    /**
     * \return The nested matrix, potentially converted to self-adjoint form.
     */
    decltype(auto) get_self_adjoint_nested_matrix() &
    {
      return this->get_nested_matrix_impl<hermitian_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_self_adjoint_nested_matrix() const &
    {
      return this->get_nested_matrix_impl<hermitian_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_self_adjoint_nested_matrix() &&
    {
      return std::move(*this).template get_nested_matrix_impl<hermitian_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_self_adjoint_nested_matrix() const &&
    {
      return std::move(*this).template get_nested_matrix_impl<hermitian_matrix<NestedMatrix>>();
    }


    /**
     * \return The nested matrix, potentially converted to triangular form.
     */
    decltype(auto) get_triangular_nested_matrix() &
    {
      return this->get_nested_matrix_impl<triangular_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_triangular_nested_matrix() const &
    {
      return this->get_nested_matrix_impl<triangular_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_triangular_nested_matrix() &&
    {
      return std::move(*this).template get_nested_matrix_impl<triangular_matrix<NestedMatrix>>();
    }


    /// \overload
    decltype(auto) get_triangular_nested_matrix() const &&
    {
      return std::move(*this).template get_nested_matrix_impl<triangular_matrix<NestedMatrix>>();
    }


    /**
     * \return The determinant.
     */
    auto determinant() const
    {
      if constexpr (triangular_covariance<Derived> and not triangular_matrix<NestedMatrix>)
      {
        if (synchronization_direction() > 0)
        {
          return square_root(OpenKalman::determinant(nested_matrix()));
        }
        else
        {
          return OpenKalman::determinant(cholesky_nested_matrix());
        }
      }
      else if constexpr (self_adjoint_covariance<Derived> and not hermitian_matrix<NestedMatrix>)
      {
        if (synchronization_direction() > 0)
        {
          auto d = OpenKalman::determinant(nested_matrix());
          return d * d;
        }
        else
        {
          return OpenKalman::determinant(cholesky_nested_matrix());
        }
      }
      else
      {
        if (synchronization_direction() < 0)
        {
          if constexpr (triangular_covariance<Derived>)
          {
            return square_root(OpenKalman::determinant(cholesky_nested_matrix()));
          }
          else
          {
            auto d = OpenKalman::determinant(cholesky_nested_matrix());
            return d * d;
          }
        }
        else
        {
          return OpenKalman::determinant(nested_matrix());
        }
      }
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementAccessor object.
     */
#ifdef __cpp_concepts
    auto operator() (std::size_t i, std::size_t j)
    requires element_settable<NestedMatrix, 2> and element_gettable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 2> and element_gettable<T, 2>, int> = 0>
    auto operator() (std::size_t i, std::size_t j)
#endif
    {
      return Base::operator()(i, j);
    }


    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i, std::size_t j) const requires element_gettable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 2>, int> = 0>
    auto operator() (std::size_t i, std::size_t j) const
#endif
    {
      return Base::operator()(i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \return An ElementAccessor object.
     */
#ifdef __cpp_concepts
    auto operator[] (std::size_t i) requires element_settable<NestedMatrix, 1> and element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 1> and element_gettable<T, 1>, int> = 0>
    auto operator[] (std::size_t i)
#endif
    {
      return Base::operator[](i);
    }


    /// \overload
#ifdef __cpp_concepts
    auto operator[] (std::size_t i) const requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator[] (std::size_t i) const
#endif
    {
      return Base::operator[](i);
    }


    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i) requires element_settable<NestedMatrix, 1> and element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 1> and element_gettable<T, 1>, int> = 0>
    auto operator() (std::size_t i)
#endif
    {
      return Base::operator[](i);
    }


    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i) const
    requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator() (std::size_t i) const
#endif
    {
      return Base::operator[](i);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
#ifdef __cpp_concepts
    void set_element(const Scalar s, const std::size_t i, const std::size_t j) requires element_settable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 2>, int> = 0>
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
#endif
    {
      Base::set_element(s, i, j);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
#ifdef __cpp_concepts
    void set_element(const Scalar s, const std::size_t i) requires element_settable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 1>, int> = 0>
    void set_element(const Scalar s, const std::size_t i)
#endif
    {
      Base::set_element(s, i);
    }


  };


} // OpenKalman::internal

#endif //OPENKALMAN_COVARIANCEIMPL_HPP
